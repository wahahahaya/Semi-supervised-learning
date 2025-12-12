# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool

def ova_loss(logits_open, label):
    """
    One-vs-All Loss for labeled data.
    logits_open: (B, 2, C) where dim 1 is (negative, positive) probability
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    
    # Create one-hot like targets for OVA
    # label_s_sp[i, c] = 1 if sample i has class c
    label_s_sp = torch.zeros((logits_open.size(0), logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    
    # Positive loss: minimize -log(prob_positive) for the correct class
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    
    # Negative loss: minimize -log(prob_negative) for all incorrect classes
    # Note: LAMDA implementation uses max optimization for negative classes in some versions, 
    # here we follow the provided code: mean(max(-log(prob_neg)))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    
    Lo = open_loss_neg + open_loss
    return Lo

def ova_ent(logits_open):
    """
    Open-Set Entropy Minimization
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    # Entropy sum over all C binary classifiers
    Le = torch.mean(torch.mean(torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1))
    return Le

@ALGORITHMS.register('openmatch')
class OpenMatch(AlgorithmBase):
    """
    OpenMatch algorithm ported from LAMDA_SSL.
    It adds an One-vs-All (OVA) head for open-set detection/filtering.
    
    Args:
        - lambda_oem: Weight for OVA Entropy Minimization
        - lambda_socr: Weight for Soft Consistency Regularization
        - start_fix_ratio: Ratio of total epochs to wait before starting FixMatch loss
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        
        self.init(T=args.T, 
                  p_cutoff=args.p_cutoff, 
                  lambda_oem=args.lambda_oem,
                  lambda_socr=args.lambda_socr,
                  start_fix_ratio=args.start_fix_ratio)
        
        # 1. Detect Feature Dimension
        if hasattr(self.model, 'num_features'):
            feat_dim = self.model.num_features
        else:
            print("Warning: Could not detect num_features, assuming 512.")
            feat_dim = 512 

        # 2. Patch Main Model
        self.ova_head = nn.Linear(feat_dim, self.num_classes * 2).to(self.gpu)
        if not hasattr(self.model, 'ova_head'):
            self.model.add_module('ova_head', self.ova_head)
        
        # 3. Patch EMA Model (=== 關鍵修復 Fix for EMA ===)
        # 檢查是否存在 EMA Model，如果存在，必須同步加上 ova_head
        if hasattr(self, 'ema_model') and self.ema_model is not None:
            # 建立一個獨立的 Linear 層給 EMA (不能跟主模型共用同一個物件)
            ova_head_ema = nn.Linear(feat_dim, self.num_classes * 2).to(self.gpu)
            
            # 初始化權重讓它跟主模型一開始一樣 (雖然之後 EMA 會自己更新，但這樣比較安全)
            ova_head_ema.load_state_dict(self.ova_head.state_dict())
            
            # 把這個層掛載到 EMA 模型上
            if not hasattr(self.ema_model, 'ova_head'):
                self.ema_model.add_module('ova_head', ova_head_ema)
                self.print_fn("OpenMatch: Patched EMA model with ova_head.")

        # Move model to GPU to ensure new parameters are on device
        self.model.to(self.gpu)


    def init(self, T, p_cutoff, lambda_oem, lambda_socr, start_fix_ratio):
        self.T = T
        self.p_cutoff = p_cutoff
        self.lambda_oem = lambda_oem
        self.lambda_socr = lambda_socr
        self.start_fix_ratio = start_fix_ratio

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        # 1. Inference - Forward pass to get Features and Closed-Set Logits
        # We concatenate everything to save forward passes (Standard USB practice)
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        
        inputs = torch.cat([x_lb, x_ulb_w, x_ulb_s], dim=0)
        outputs = self.model(inputs)
        
        logits = outputs['logits']
        feats = outputs['feat']
        
        # 2. Compute OVA Logits using our extra head
        # output shape: (B, 2*C) -> reshape to (B, 2, C) inside loss functions
        logits_open = self.model.ova_head(feats)
        
        # Slicing
        logits_x_lb = logits[:num_lb]
        logits_open_lb = logits_open[:num_lb]
        
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        logits_open_ulb_w, logits_open_ulb_s = logits_open[num_lb:].chunk(2)
        
        # 3. Supervised Losses
        # 3.1 Standard CE
        sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
        
        # 3.2 OVA Loss (Labeled)
        ova_sup_loss = ova_loss(logits_open_lb, y_lb)

        # 4. Open-Set Losses
        # 4.1 OEM: Open-set Entropy Minimization on Weak Unlabeled
        L_oem = ova_ent(logits_open_ulb_w) / 2.0
        L_oem += ova_ent(logits_open_ulb_s) / 2.0  # Also use strong if available
        
        # 4.2 SOCR: Soft Consistency Regularization
        # Minimize distance between OVA predictions of Weak and Strong
        l_open_w_view = logits_open_ulb_w.view(logits_open_ulb_w.size(0), 2, -1)
        l_open_s_view = logits_open_ulb_s.view(logits_open_ulb_s.size(0), 2, -1)
        
        prob_open_w = F.softmax(l_open_w_view, 1)
        prob_open_s = F.softmax(l_open_s_view, 1)
        
        L_socr = torch.mean(torch.sum(torch.sum(torch.abs(prob_open_w - prob_open_s)**2, 1), 1))

        # 5. FixMatch Loss (Closed-Set)
        # Only apply after warmup period
        current_iter = self.it
        total_iter = self.num_train_iter
        
        # LAMDA logic: if self.it_total >= self.start_fix*self.num_it_total:
        if current_iter >= self.start_fix_ratio * total_iter:
            # Standard FixMatch logic
            with torch.no_grad():
                probs_x_ulb_w = torch.softmax(logits_x_ulb_w / self.T, dim=-1)
                max_probs, targets_u = torch.max(probs_x_ulb_w, dim=-1)
                mask = max_probs.ge(self.p_cutoff).float()
            
            # Cross Entropy on Strong Augmentation
            L_fix = (F.cross_entropy(logits_x_ulb_s, targets_u, reduction='none') * mask).mean()
        else:
            L_fix = torch.tensor(0.0).to(self.gpu)
            mask = torch.tensor(0.0).to(self.gpu) # Just for logging

        # 6. Total Loss
        total_loss = sup_loss + ova_sup_loss + \
                     self.lambda_oem * L_oem + \
                     self.lambda_socr * L_socr + \
                     L_fix

        out_dict = self.process_out_dict(loss=total_loss, feat=feats)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         ova_sup_loss=ova_sup_loss.item(),
                                         oem_loss=L_oem.item(),
                                         socr_loss=L_socr.item(),
                                         fix_loss=L_fix.item(),
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 1.0),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--lambda_oem', float, 0.1),
            SSL_Argument('--lambda_socr', float, 0.5),
            SSL_Argument('--start_fix_ratio', float, 0.02, help='Ratio of iterations to warmup OVA before FixMatch'),
        ]