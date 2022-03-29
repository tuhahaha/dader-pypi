import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.image import imread
from IPython import display
from tqdm.notebook import tqdm
import sys
sys.path.append("..")
from ..metrics import coral
from . import evaluate

def adapt_coral(encoder, matcher, alignment, 
                src_data_loader, tgt_data_loader, valid_data_loader, 
                lr=1e-5, epochs=40, device='cpu', alpha=1, beta=0.01, **kwargs):
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    encoder.train()
    matcher.train()
    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters())+list(matcher.parameters()), lr=lr)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    y = []
    corals = []
    clss = []
    f1_score = []
    
    if valid_data_loader != None:
        bestf1 = 0.0
        f1_valid = 0.0
        best_encoder = copy.deepcopy(encoder)
        best_matcher = copy.deepcopy(matcher)
        best_alignment = copy.deepcopy(alignment)

    for epoch in range(epochs):
        if len(src_data_loader)>len(tgt_data_loader):
            data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        else:
            data_zip = enumerate(zip(src_data_loader, tgt_data_loader))

        for step, (src, tgt) in data_zip:
            if tgt:
                seq_src, src_mask,src_segment, labels, _ = src
                seq_tgt, tgt_mask,tgt_segment, _, _ = tgt
                seq_src = seq_src.to(device)
                src_mask = src_mask.to(device)
                src_segment = src_segment.to(device)
                labels = labels.to(device)
                seq_tgt = seq_tgt.to(device)
                tgt_mask = tgt_mask.to(device)
                tgt_segment = tgt_segment.to(device)
    
                # zero gradients for optimizer
                optimizer.zero_grad()
    
                # extract and concat features
                feat_src = encoder(seq_src, src_mask, src_segment)
                feat_tgt = encoder(seq_tgt, tgt_mask, tgt_segment)
                preds = matcher(feat_src)
                cls_loss = CELoss(preds, labels)
                loss_coral = coral.cal_coral_loss(feat_src, feat_tgt)
                # p = float(step + epoch * len_data_loader) / epochs / len_data_loader
                # lamda = 2. / (1. + np.exp(-10 * p)) - 1
                
                loss = alpha * cls_loss + beta * loss_coral
            else:
                seq_src, src_mask,src_segment, labels = src
                seq_src = seq_src.to(device)
                src_mask = src_mask.to(device)
                src_segment = src_segment.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
    
                # extract and concat features
                feat_src = encoder(seq_src, src_mask, src_segment)
                preds = matcher(feat_src)
                cls_loss = CELoss(preds, labels)
                # p = float(step + epoch * len_data_loader) / epochs / len_data_loader
                # lamda = 2. / (1. + np.exp(-10 * p)) - 1

                loss = cls_loss
            loss.backward()
            optimizer.step()
        
            if (step + 1) % 100 == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "coral_loss=%.4f cls_loss=%.4f"
                      % (epoch + 1,
                         epochs,
                         step + 1,
                         len_data_loader,
                         loss_coral.item(),
                         cls_loss.item()
                         ))
        
        if valid_data_loader != None:
            f1_valid = evaluate(encoder, matcher, valid_data_loader, device)
            if f1_valid>bestf1:
                print("best epoch number: ",epoch)
                print("best F1-Score: ",f1_valid)
                bestf1 = f1_valid

                best_encoder = copy.deepcopy(encoder)
                best_matcher = copy.deepcopy(matcher)
                best_alignment = copy.deepcopy(alignment)
                
    if valid_data_loader == None:
        return encoder,matcher,alignment
    else:
        return best_encoder,best_matcher,best_alignment
