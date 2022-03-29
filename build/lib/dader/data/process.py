import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm.notebook import tqdm

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids=None, input_mask=None, segment_ids=None,label_id=None,exm_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.exm_id = exm_id

class InputFeaturesED(object):
    """A single set of features of data for ED."""
    def __init__(self, input_ids, attention_mask,label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id

def convert_examples_to_features(pairs, labels, max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0):
   # print("convert %d examples to features" % len(pairs))
    features = []
    if labels == None:
        labels = [0]*len(pairs)
    datazip =list(zip(pairs,labels))
    for i in range(len(datazip)):
        (pair, label) = datazip[i]
        #if (ex_index + 1) % 200 == 0:
         #   print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        # [CLS] seq1 [SEP] seq2 [SEP]
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more <len(rtokens) : # remove excessively long string
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("The sequence is too long, please add the ``max_seq_length``!")
                    continue
            tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]
            segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)
        # [CLS] seq1 [SEP]
        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [0]*(len(tokens))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids = segment_ids,
                          label_id=label,
                          exm_id=i))
    return features

def convert_examples_to_features_ED(pairs, labels, max_seq_length, tokenizer, 
                                    pad_token=0, cls_token='<s>',sep_token='</s>'):
    features = []
    if labels == None:
        labels = [0]*len(pairs)
    for ex_index, (pair, label) in enumerate(zip(pairs, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more <len(rtokens) :
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("The sequence is too long, please add the ``max_seq_length``!")
                    continue
            tokens =  [cls_token] +ltokens + [sep_token] + rtokens + [sep_token]
            segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)
        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [0]*(len(tokens))
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        features.append(InputFeaturesED(input_ids=input_ids,
                        attention_mask=input_mask,
                        label_id=label
                        ))
    return features

def get_data_loader(features, batch_size, is_train=0):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_exm_ids = torch.tensor([f.exm_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_label_ids,all_exm_ids)
    
    if is_train:
        """Delet the last incomplete epoch"""
        # sampler = RandomSampler(dataset)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    else:
        """Read all data"""
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

def get_data_loader_ED(features, batch_size, is_train=0):
    """
    data_loader for Reconstruction-based method
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,all_label_ids)
    
    if is_train:
        """Delet the last incomplete epoch"""
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    else:
        """Read all data"""
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

