import os
import random
import pandas as pd 
from IPython import display
import torch
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer
from transformers import logging
from . import BertEncoder, BartEncoder
from . import BertMatcher
from . import Discriminator, DomainClassifier, BartDecoder
import sys
sys.path.append("..")
from ..data import load_data, get_data_loader, get_data_loader_ED, \
                    convert_examples_to_features, convert_examples_to_features_ED
from ..runner import pretrain, evaluate, eval_predict, write_encode_to_csv, show_retrain_result, \
                    adapt_mmd, adapt_coral, adapt_grl, adapt_invgan, adapt_invgan_kd, adapt_ed
from .. import param

"""
method: [noda, mmd, koral, grl, invgan, invgankd, ed]
architecture: [Bert, RoBERTa, Bart]
load: load saved model from checkpoints

pretrain: whether or not train fe and matcher first with Source
adapt: whether or not adapt model with Target
alpha,beta: loss weight

"""
logging.set_verbosity_warning()

class Model():
    def __init__(self, method='mmd', architecture='Bert', tokenizer=None, \
                    encoder=None, aligner=None, matcher=None,
                    hidden_size=768, num_labels=2, seed=31, **kwargs):
        self.method = method

        self.encoder = encoder
        self.aligner = aligner
        self.matcher = matcher
        self.tokenizer = tokenizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.res_results = []
        self.set_seed(seed)

        if architecture == None and (self.encoder == None or self.matcher == None):
            print("architecture, encoder, matcher can't be None concurrently! \n \
                    We will use the default settings! ")
            if method != "ed":
                architecture = 'Bert'
            else:
                architecture = 'Bart'

        if architecture == 'Bert':
            self.encoder = BertEncoder()
            self.matcher = BertMatcher(hidden_size=hidden_size, num_labels=num_labels)
            self.tokenizer = BertTokenizer.from_pretrained(param.default_bert)
        elif architecture == 'Bart':
            self.encoder = BartEncoder()
            self.matcher = BertMatcher(hidden_size=hidden_size, num_labels=num_labels)
            self.tokenizer = BartTokenizer.from_pretrained(param.default_bart)

        # Setting aligner according to method
        if not self.aligner and self.method in ['invgan','invgankd']:
            self.aligner = Discriminator(hidden_size) # we just give one choice for aligner
        elif not self.aligner and self.method == 'grl':
            self.aligner = DomainClassifier(hidden_size) # we just give one choice for aligner
        elif not self.aligner and self.method == 'ed':
            self.aligner = BartDecoder() # we just give one choice for aligner
        
        self.encoder.to(self.device)
        self.matcher.to(self.device)
        if self.aligner:
            self.aligner.to(self.device)
        display.clear_output(wait=True)

    def set_seed(self,seed):
        random.seed(seed)
        torch.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(seed)


    def fit(self, 
                src_x = None,
                src_y = None,
                tgt_x = None,
                valid_x = None,
                valid_y = None,
                tgt_y = None,
                pre=True,
                adapt=True,
                load=False,
                max_seq_length=128,
                batch_size=32,
                pre_max_epoch=3,
                ada_max_epoch=10,
                pre_lr=2e-5,
                ada_lr=1e-6,
                alpha=1,
                beta=5,
                max_grad_norm=1.0,
                clip_value=0.01, 
                temperature=20,
                seed=31,
                train_seed=31,
                 **kwargs):
        """
        pretrain + adapt
        """
        # self.set_seed(train_seed)

        # initial networks
        if load:
            self.encoder = load_model(self.encoder, restore=param.encoder_path)
            self.matcher = load_model(self.matcher, restore=param.encoder_path)
            self.aligner = load_model(self.aligner, restore=param.encoder_path)
        self.encoder.to(self.device)
        self.matcher.to(self.device)
        if self.aligner:
            self.aligner.to(self.device)
        
        if tgt_y == None:
            tgt_y = [0]*len(tgt_x)

        # convert data
        if self.method == 'ed':
            src_features = convert_examples_to_features_ED(src_x, src_y, max_seq_length, self.tokenizer)
            src_data_loader = get_data_loader_ED(src_features, batch_size, is_train=1)
            tgt_features = convert_examples_to_features_ED(tgt_x, tgt_y, max_seq_length, self.tokenizer)
            tgt_data_loader = get_data_loader_ED(tgt_features, batch_size, is_train=1)
            valid_data_loader = None
            if valid_x:
                valid_features = convert_examples_to_features_ED(valid_x, valid_y, max_seq_length, self.tokenizer)
                valid_data_loader = get_data_loader_ED(valid_features, batch_size, is_train=0)
        else:
            src_features = convert_examples_to_features(src_x, src_y, max_seq_length, self.tokenizer)
            src_data_loader = get_data_loader(src_features, batch_size, is_train=1)
            tgt_features = convert_examples_to_features(tgt_x, tgt_y, max_seq_length, self.tokenizer)
            tgt_data_loader = get_data_loader(tgt_features, batch_size, is_train=1)
            valid_data_loader = None
            if valid_x:
                valid_features = convert_examples_to_features(valid_x, valid_y, max_seq_length, self.tokenizer)
                valid_data_loader = get_data_loader(valid_features, batch_size, is_train=0)


        # pretrain encoder + matcher
        if pre:
            print("Pretraining...")
            self.encoder, self.matcher = pretrain(self.encoder, self.matcher, 
                                                src_data_loader, valid_data_loader,
                                                lr=pre_lr, epochs=pre_max_epoch, device=self.device)
            res = evaluate(self.encoder, self.matcher, tgt_data_loader, self.device, is_out=False)

        # adapt
        if adapt:
            if self.method == "mmd":
                self.encoder, self.matcher, self.aligner = adapt_mmd(self.encoder, self.matcher, self.aligner,
                                            src_data_loader, tgt_data_loader, valid_data_loader,
                                            lr=ada_lr, epochs=ada_max_epoch, device=self.device,
                                            alpha=alpha, beta=beta)
            elif self.method == "coral":
                self.encoder, self.matcher, self.aligner = adapt_coral(self.encoder, self.matcher, self.aligner,
                                            src_data_loader, tgt_data_loader, valid_data_loader,
                                            lr=ada_lr, epochs=ada_max_epoch, device=self.device,
                                            alpha=alpha, beta=beta)
            elif self.method == "grl":
                self.encoder, self.matcher, self.aligner = adapt_grl(self.encoder, self.matcher, self.aligner,
                                            src_data_loader, tgt_data_loader, valid_data_loader,
                                            lr=ada_lr, epochs=ada_max_epoch, device=self.device,
                                            alpha=alpha, beta=beta)
            elif self.method == "invgan":
                self.encoder, self.matcher, self.aligner = adapt_invgan(self.encoder, self.matcher, self.aligner,
                                            src_data_loader, tgt_data_loader, valid_data_loader,
                                            lr=ada_lr, epochs=ada_max_epoch, device=self.device,
                                            alpha=alpha, beta=beta, 
                                            clip_value=clip_value, max_grad_norm=max_grad_norm)
            elif self.method == "invgankd":
                self.encoder, self.matcher, self.aligner = adapt_invgan_kd(self.encoder, self.matcher, self.aligner,
                                            src_data_loader, tgt_data_loader, valid_data_loader,
                                            lr=ada_lr, epochs=ada_max_epoch, device=self.device,
                                            alpha=alpha, beta=beta, 
                                            clip_value=clip_value, temperature=temperature, 
                                            max_grad_norm=max_grad_norm)
                                            
            elif self.method == "ed":
                self.encoder, self.matcher, self.aligner = adapt_ed(self.encoder, self.matcher, self.aligner,
                                            src_data_loader, tgt_data_loader, valid_data_loader,
                                            lr=ada_lr, epochs=ada_max_epoch, device=self.device,
                                            alpha=alpha, beta=beta)
            else:
                print("Please choose one method from [mmd, coral, grl, invgan, invgankd, ed].")
                exit()

        # evaluate on target dataset
        recall, precision, f1 = evaluate(self.encoder, self.matcher, tgt_data_loader, self.device, is_out=True)

    
    def finetune(self, 
                src_x = None,
                src_y = None,
                tgt_x = None,
                valid_x = None,
                valid_y = None,
                tgt_y = None,
                load=False,
                max_seq_length=128,
                batch_size=32,
                pre_max_epoch=3,
                pre_lr=2e-5,
                max_grad_norm=1.0,
                clip_value=0.01, 
                seed=42,
                train_seed=10,
                 **kwargs):
        """
        pretrain + adapt
        """
        # self.set_seed(train_seed)
    
        if tgt_y == None:
            tgt_y = [0]*len(tgt_x)
        # initial networks
        if load:
            self.encoder = load_model(self.encoder, restore=param.encoder_path)
            self.matcher = load_model(self.matcher, restore=param.encoder_path)
            self.aligner = load_model(self.aligner, restore=param.encoder_path)
        self.encoder.to(self.device)
        self.matcher.to(self.device)
        if self.aligner:
            self.aligner.to(self.device)

        # convert data
        if self.method == 'ed':
            src_features = convert_examples_to_features_ED(src_x, src_y, max_seq_length, self.tokenizer)
            src_data_loader = get_data_loader_ED(src_features, batch_size, is_train=1)
            tgt_features = convert_examples_to_features_ED(tgt_x, tgt_y, max_seq_length, self.tokenizer)
            tgt_data_loader = get_data_loader_ED(tgt_features, batch_size, is_train=1)
            valid_data_loader = None
            if valid_x:
                valid_features = convert_examples_to_features_ED(valid_x, valid_y, max_seq_length, self.tokenizer)
                valid_data_loader = get_data_loader_ED(valid_features, batch_size, is_train=0)
        else:
            src_features = convert_examples_to_features(src_x, src_y, max_seq_length, self.tokenizer)
            src_data_loader = get_data_loader(src_features, batch_size, is_train=1)
            tgt_features = convert_examples_to_features(tgt_x, tgt_y, max_seq_length, self.tokenizer)
            tgt_data_loader = get_data_loader(tgt_features, batch_size, is_train=1)
            valid_data_loader = None
            if valid_x:
                valid_features = convert_examples_to_features(valid_x, valid_y, max_seq_length, self.tokenizer)
                valid_data_loader = get_data_loader(valid_features, batch_size, is_train=0)


        # pretrain encoder + matcher
        self.encoder, self.matcher = pretrain(self.encoder, self.matcher, 
                                                src_data_loader, valid_data_loader,
                                                lr=pre_lr, epochs=pre_max_epoch, device=self.device)
        
        
        # evaluate on target dataset
        recall, precision, f1 = evaluate(self.encoder, self.matcher, tgt_data_loader, self.device, is_out=True)
    
    def predict(self, x=None, y=None, max_seq_length=128, batch_size=32, **kwargs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.method == 'ed':
            features = convert_examples_to_features_ED(x, y, max_seq_length, self.tokenizer)
            data_loader = get_data_loader_ED(features, batch_size, is_train=0)
            # predict and save to predict_prob.csv
            file_name = eval_predict_ED(self.encoder, self.matcher, data_loader, device)

        else:
            features = convert_examples_to_features(x, y, max_seq_length, self.tokenizer)
            data_loader = get_data_loader(features, batch_size, is_train=0)
            # predict and save to predict_prob.csv
            file_name, predict = eval_predict(self.encoder, self.matcher, data_loader, device)

        #print("The predicted results have been saved to ",file_name)
        return predict
    
    def eval(self, X_tgt, y_prd, y_tgt, max_seq_length=128 , batch_size=32):
        tgt_features = convert_examples_to_features(X_tgt, y_tgt, max_seq_length, self.tokenizer)
        tgt_data_loader = get_data_loader(tgt_features, batch_size, is_train=1)
        recall, precision, f1 = evaluate(self.encoder, self.matcher, tgt_data_loader, self.device, is_out=True)
        self.res_results.append([recall, precision, f1])
        if len(self.res_results)==1:
            df = pd.DataFrame(self.res_results,index=['After DA'],columns=['recall','precision','F1'])
        else:
            index = ['After DA'] + ['After Retrain']*(len(self.res_results)-1)
            df = pd.DataFrame(self.res_results,index=index,columns=['recall','precision','F1'])
            #show_retrain_result(self.res_results)
        
        return df


    def load_model(self, net, name=None):
        if name is not None:
            path = os.path.join(param.model_root, name)
            if os.path.exists(path):
                net.load_state_dict(torch.load(path))
                print("Restore model from: {}".format(os.path.abspath(path)))
        return net


    def save_model(self, net, name):
        """Save trained model."""
        folder = param.model_root
        path = os.path.join(folder, name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(net.state_dict(), path)
        print("save pretrained model to: {}".format(path))





