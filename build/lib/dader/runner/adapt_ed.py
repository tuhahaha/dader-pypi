import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import copy
from . import evaluate_ED

def adapt_ed(encoder, matcher, decoder,
                src_data_loader, tgt_data_loader, valid_data_loader,
                lr=1e-5, epochs=40, device='cpu', alpha=1, beta=0.01, **kwargs):
    """Train encoder for target domain."""
    # set train state for Dropout and BN layers
    encoder.train()
    matcher.train()
    decoder.train()
    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    CELoss = CrossEntropyLoss()
    optimizer1 = optim.Adam(list(encoder.parameters())+list(matcher.parameters()), lr=lr)
    optimizer2 = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=lr)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    
    if valid_data_loader != None:
        bestf1 = 0.0
        best_encoder = copy.deepcopy(encoder)
        best_matcher = copy.deepcopy(matcher)
        best_decoder = copy.deepcopy(decoder)

    for epoch in range(epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((seq_src, src_mask,labels), (seq_tgt, tgt_mask,_)) in data_zip:
            seq_src = seq_src.to(device)
            src_mask = src_mask.to(device)
            labels = labels.to(device)
            seq_tgt = seq_tgt.to(device)
            tgt_mask = tgt_mask.to(device)

            # zero gradients for optimizer
            optimizer1.zero_grad()
            
            # train encoder and matcher
            encoder_output,logit_,pooled_output = encoder(seq_src, src_mask)
            vocab_size=decoder.config.vocab_size
            preds = matcher(pooled_output)
  
            cls_loss = CELoss(preds, labels)
            loss = alpha * cls_loss
            loss.backward()
            optimizer1.step()
            
            if 1: # source
                optimizer2.zero_grad()
                encoder_outputs,_,_1=encoder(seq_tgt,tgt_mask)
                decoder_outputs,logits1=decoder(seq_tgt, encoder_hidden_states=encoder_outputs[0],attention_mask=tgt_mask)                
                hidden_size=decoder.config.hidden_size
                loss1=CELoss(logits1.view(-1,vocab_size),seq_tgt.view(-1))
    
                loss1.backward()
                optimizer2.step()
            if 1: # target
                optimizer2.zero_grad()
                encoder_outputs,_,_1=encoder(seq_src,src_mask)
                decoder_outputs,logits1=decoder(seq_src, encoder_hidden_states=encoder_outputs[0],attention_mask=src_mask)
                loss1=CELoss(logits1.view(-1,vocab_size),seq_src.view(-1))
                loss1.backward()
                optimizer2.step()
                
              
            if (step + 1) % 100 == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "cls_loss=%.4f,loss=%.4f"
                      % (epoch + 1,
                         epochs,
                         step + 1,
                         len_data_loader,
                         loss.item(),
                         loss1.item()))        

        if valid_data_loader != None:
            f1_valid = evaluate_ED(encoder, matcher, valid_data_loader, device)
            if f1_valid>bestf1:
                print("best epoch number: ",epoch)
                print("best F1-Score: ",f1_valid)
                bestf1 = f1_valid

                best_encoder = copy.deepcopy(encoder)
                best_matcher = copy.deepcopy(matcher)
                best_decoder = copy.deepcopy(decoder)
                
    if valid_data_loader == None:
        return encoder,matcher,decoder
    else:
        return best_encoder,best_matcher,best_decoder

