import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import copy
from . import evaluate


def adapt_invgan_kd(encoder, matcher, alignment, 
                src_data_loader, tgt_data_loader, valid_data_loader,
                lr=1e-5, epochs=40, device='cpu', alpha=1, beta=1, 
                clip_value=0.01, temperature=20, max_grad_norm=1.0, **kwargs):
    """INvGAN+KD without valid data"""

    # set train state for Dropout and BN layers
    tgt_encoder = copy.deepcopy(encoder)
    encoder.eval()
    matcher.eval()
    for params in encoder.parameters():
        params.requires_grad = False
    for params in matcher.parameters():
        params.requires_grad = False

    tgt_encoder.train()
    alignment.train()

    # setup criterion and optimizer
    BCELoss = nn.BCELoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    optimizer_G = optim.Adam(tgt_encoder.parameters(), lr=lr)
    optimizer_D = optim.Adam(alignment.parameters(), lr=lr)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    if valid_data_loader != None:
        bestf1 = 0.0
        f1_valid = 0.0
        best_encoder = copy.deepcopy(tgt_encoder)
        best_matcher = copy.deepcopy(matcher)
        best_alignment = copy.deepcopy(alignment)

    for epoch in range(epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((seq_src, src_mask,src_segment, _, _), (seq_tgt, tgt_mask,tgt_segment, _, _)) in data_zip:
            seq_src = seq_src.to(device)
            src_mask = src_mask.to(device)
            src_segment = src_segment.to(device)

            seq_tgt = seq_tgt.to(device)
            tgt_mask = tgt_mask.to(device)
            tgt_segment = tgt_segment.to(device)

            # zero gradients for optimizer
            optimizer_D.zero_grad()

            # extract and concat features
            with torch.no_grad():
                feat_src = encoder(seq_src, src_mask,src_segment)
            feat_src_tgt = tgt_encoder(seq_src, src_mask,src_segment)
            feat_tgt = tgt_encoder(seq_tgt, tgt_mask,tgt_segment)
            feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)

            # predict on alignment
            pred_concat = alignment(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(feat_src_tgt.size(0)).to(device).unsqueeze(1)
            label_tgt = torch.zeros(feat_tgt.size(0)).to(device).unsqueeze(1)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for alignment
            dis_loss = BCELoss(pred_concat, label_concat)
            dis_loss.backward()

            for p in alignment.parameters():
                p.data.clamp_(-clip_value, clip_value)
            # optimize alignment
            optimizer_D.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            # zero gradients for optimizer
            optimizer_G.zero_grad()
            T = temperature

            # predict on alignment
            pred_tgt = alignment(feat_tgt)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = F.softmax(matcher(feat_src) / T, dim=-1)
            tgt_prob = F.log_softmax(matcher(feat_src_tgt) / T, dim=-1)
            kd_loss = KLDivLoss(tgt_prob, src_prob.detach()) * T * T

            # compute loss for target encoder
            gen_loss = BCELoss(pred_tgt, label_src)
            loss_tgt = alpha * gen_loss + beta * kd_loss
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), max_grad_norm)
            # optimize target encoder
            optimizer_G.step()
            
            if (step + 1) % 10 == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "acc=%.4f g_loss=%.4f d_loss=%.4f kd_loss=%.4f"
                      % (epoch + 1,
                         epochs,
                         step + 1,
                         len_data_loader,
                         acc.item(),
                         gen_loss.item(),
                         dis_loss.item(),
                         kd_loss.item()))
          
        # print("src M on target:")
        # evaluate(encoder, matcher, tgt_data_loader, device)
        # print("tgt M on target:")
        #evaluate(tgt_encoder, matcher, tgt_data_loader, device)
        if valid_data_loader != None:
            f1_valid = evaluate(tgt_encoder, matcher, valid_data_loader, device)
            if f1_valid>bestf1:
                print("best epoch number: ",epoch)
                print("best F1-Score: ",f1_valid)
                bestf1 = f1_valid

                best_encoder = copy.deepcopy(tgt_encoder)
                best_matcher = copy.deepcopy(matcher)
                best_alignment = copy.deepcopy(alignment)

    if valid_data_loader == None:
        return tgt_encoder,matcher,alignment
    else:
        return best_encoder,best_matcher,best_alignment

