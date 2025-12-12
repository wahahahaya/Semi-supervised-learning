# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import numpy as np
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

@ALGORITHMS.register('fix_a_step')
class Fix_A_Step(AlgorithmBase):
    """
    Fix_A_Step algorithm ported from LAMDA_SSL.
    It combines MixUp (from MixMatch) with a Gradient-based dynamic weight adjustment mechanism.
    
    Args:
        - args (`argparse`): algorithm arguments
        - net_builder (`callable`): network loading function
        - tb_log (`TBLog`): tensorboard logger
        - logger (`logging.Logger`): logger to use
        - T (`float`): Temperature for pseudo-label sharpening
        - p_cutoff(`float`): Confidence threshold for generating pseudo-labels
        - alpha (`float`): Beta distribution parameter for MixUp
        - warmup_epochs (`float`): Epochs to wait before applying the gradient check strategy
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fix_a_step specified arguments
        # 使用 getattr 來避免 AttributeError，預設值設為 True 或 0.75 等
        self.init(T=args.T, 
                  p_cutoff=args.p_cutoff, 
                  hard_label=getattr(args, 'hard_label', True),        # 修改這行
                  alpha=getattr(args, 'mixup_alpha', 0.75),            # 修改這行 (也加上安全措施)
                  warmup_epochs=getattr(args, 'warmup_epochs', 0.0))   # 修改這行 (也加上安全措施)

    def init(self, T, p_cutoff, hard_label=True, alpha=0.75, warmup_epochs=0):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = x_lb.shape[0]

        # 1. Pseudo-Label Generation (No Grad)
        with torch.no_grad():
            self.model.eval()
            # USB typically gives one weak augmentation. 
            # LAMDA code uses two weak views, here we use the available x_ulb_w.
            outputs_w = self.model(x_ulb_w)
            logits_u_w = outputs_w['logits']
            
            # Sharpening
            p = torch.softmax(logits_u_w, dim=1)
            pt = p**(1/self.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
            
            # Masking for reporting usage later
            max_probs, _ = torch.max(p, dim=-1)
            mask = max_probs.ge(self.p_cutoff).float()
            
            self.model.train()

        # 2. MixUp Preparation
        # Transform labels to one-hot
        y_lb_onehot = torch.zeros(num_lb, self.num_classes).to(self.gpu).scatter_(1, y_lb.view(-1, 1).long(), 1)
        
        # Combine inputs and targets for MixUp
        # Note: LAMDA mixes [lb, ulb_w1, ulb_w2]. USB has [lb, ulb_w, ulb_s].
        # We will mix [lb, ulb_w, ulb_s] to utilize strong augmentation as well similar to MixMatch/FixMatch hybrids.
        inputs = torch.cat([x_lb, x_ulb_w, x_ulb_s], dim=0)
        targets = torch.cat([y_lb_onehot, targets_u, targets_u], dim=0)

        l = np.random.beta(self.alpha, self.alpha)
        l = max(l, 1-l)
        
        idx = torch.randperm(inputs.size(0))
        input_a, input_b = inputs, inputs[idx]
        target_a, target_b = targets, targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # Split back: The first batch_size is treated as labeled for the supervised loss
        mixed_lb_input = mixed_input[:num_lb]
        mixed_lb_target = mixed_target[:num_lb]
        
        # The rest are unlabeled parts (weak and strong mixed)
        # In LAMDA implementation logic:
        # inputs = interleave(cat(mixed_labeled, w_ulb, s_ulb))
        # Here mixed_input contains mixed versions of everything.
        # To strictly follow LAMDA's structure of "Supervised on Mixed Labeled" vs "Unsupervised on Raw Unlabeled":
        # LAMDA does: inputs = interleave(torch.cat((mixed_labeled_input, w_ulb_X_1, s_ulb_X)), ...)
        # So we need to compute logits for: Mixed Labeled AND Raw Unlabeled (Weak & Strong)
        
        # Re-construct batch for forward pass
        # We need gradients for parameters, so we use the model in training mode
        combined_input = torch.cat([mixed_lb_input, x_ulb_w, x_ulb_s], dim=0)
        
        # Interleave to handle Batch Normalization statistics correctly
        combined_input = interleave(combined_input, 2*self.args.mu + 1).to(self.gpu)
        
        outputs = self.model(combined_input)
        logits = de_interleave(outputs['logits'], 2*self.args.mu + 1)
        
        logits_x = logits[:num_lb]
        logits_u_w_out, logits_u_s_out = logits[num_lb:].chunk(2)

        # 3. Loss Calculation
        # Labeled Loss (on Mixed Data)
        # Softmax cross entropy with mixed targets
        sup_loss = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_lb_target, dim=1))

        # Unlabeled Loss (Standard FixMatch style: Cross Entropy on Strong Augmentation using Weak Targets)
        # Note: LAMDA uses `logits_u_s` vs `targets_u` with mask
        unsup_loss = (F.cross_entropy(logits_u_s_out, targets_u.argmax(dim=-1), reduction='none') * mask).mean()

        # 4. Fix_A_Step Gradient Logic
        # We need to inspect gradients to decide how to combine losses.
        # Since we cannot easily access optimizer inside train_step in USB, we use torch.autograd.grad
        
        # Current progress for warmup check
        current_epoch = self.epoch
        is_warmup = current_epoch < self.warmup_epochs

        final_loss = sup_loss + self.lambda_u * unsup_loss # Default fallback

        if not is_warmup:
            params = [p for p in self.model.parameters() if p.requires_grad]
            
            # 1. 計算 Labeled Gradients
            # create_graph=False (我們不需要對 dot product 微分)
            # retain_graph=True (因為 sup_loss 稍後還要被 backward 一次)
            grads_x = torch.autograd.grad(sup_loss, params, retain_graph=True, allow_unused=True)
            
            # 2. 計算 Unlabeled Gradients
            grads_u = torch.autograd.grad(unsup_loss, params, retain_graph=True, allow_unused=True)

            # 3. 扁平化並處理 None (關鍵修正)
            # 如果 g 是 None，補上全 0 的 Tensor 以保持對齊
            grads_x_flat = []
            for g, p in zip(grads_x, params):
                if g is not None:
                    grads_x_flat.append(g.view(-1))
                else:
                    grads_x_flat.append(torch.zeros_like(p).view(-1))
            grads_x_flat = torch.cat(grads_x_flat)

            grads_u_flat = []
            for g, p in zip(grads_u, params):
                if g is not None:
                    grads_u_flat.append(g.view(-1))
                else:
                    grads_u_flat.append(torch.zeros_like(p).view(-1))
            grads_u_flat = torch.cat(grads_u_flat)

            # 4. Dot Product
            gradient_dot = torch.dot(grads_x_flat, grads_u_flat)

            if gradient_dot < 0:
                final_loss = sup_loss
            else:
                final_loss = sup_loss + self.lambda_u * unsup_loss
        
        # 5. Logging and Return
        # USB Trainer will call final_loss.backward()
        
        out_dict = self.process_out_dict(loss=final_loss, feat=outputs['feat'])
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=final_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--mixup_alpha', float, 0.75, help='Alpha parameter for MixUp beta distribution'),
            SSL_Argument('--warmup_epochs', float, 0, help='Number of epochs before applying Fix_A_Step gradient check'),
        ]