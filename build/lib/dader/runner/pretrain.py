"""Pretrain F and M with labeled Source data."""
import torch.nn as nn
import torch.optim as optim
import datetime
import copy

from .evaluate import evaluate
from tqdm.notebook import tqdm


def pretrain(encoder, matcher, data_loader, valid_data_loader, lr=2e-5, epochs=3, device='cpu', **kwargs):
    """Train F and M for source domain with valid dataset."""

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(matcher.parameters()),
                           lr=lr)
    CELoss = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    matcher.train()

    if valid_data_loader != None:
        best_f1 = 0
        best_encoder = copy.deepcopy(encoder)
        best_matcher = copy.deepcopy(matcher)

    for epoch in range(epochs):
        for step, (seq, mask, segment, labels,_) in enumerate(data_loader):
            seq = seq.to(device)
            mask = mask.to(device)
            segment = segment.to(device)
            labels = labels.to(device)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for discriminator
            feat = encoder(seq, mask,segment)
            preds = matcher(feat)
            cls_loss = CELoss(preds, labels)

            # optimize source matcher
            cls_loss.backward()
            optimizer.step()

            # print step info
            if (step + 1) % 10 == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: loss=%.4f"
                      % (epoch + 1,
                         epochs,
                         step + 1,
                         len(data_loader),
                         cls_loss.item()))
        
        if valid_data_loader != None: # need to choose best model by valid dataset
            f1 = evaluate(encoder, matcher, valid_data_loader, device)
            if f1 > best_f1:
                print("best epoch number: ",epoch)
                print("best F1-Score: ",f1)
                best_f1 = f1

                # record best model
                best_encoder = copy.deepcopy(encoder)
                best_matcher = copy.deepcopy(matcher)
    
    
    if not valid_data_loader:
        return encoder, matcher
    else:
        return best_encoder, best_matcher

