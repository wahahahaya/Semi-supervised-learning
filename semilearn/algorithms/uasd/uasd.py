# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool

@ALGORITHMS.register('uasd')
class UASD(AlgorithmBase):
    """
    UASD (Uncertainty-Aware Self-Distillation) ported from LAMDA_SSL.
    
    Args:
        - args (`argparse`): algorithm arguments
        - net_builder (`callable`): network loading function
        - tb_log (`TBLog`): tensorboard logger
        - logger (`logging.Logger`): logger to use
        - num_samples (`int`): The number of unlabeled samples (auto-detected usually)
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        
        # UASD arguments
        self.init(threshold=args.p_cutoff,
                  lambda_u=args.ulb_loss_ratio)
        
        # Memory Banks (Initialized lazily in train_step)
        self.pslab = None         
        self.epoch_pslab = None   
        self.num_ulb_samples = 0
        
    def init(self, threshold, lambda_u):
        self.threshold = threshold
        self.lambda_u = lambda_u

    def set_hooks(self):
        super().set_hooks()
        # We need to initialize the memory banks. 
        # In USB, self.dataset_dict is available after initialization.
        if 'train_ulb' in self.dataset_dict:
            self.num_ulb_samples = len(self.dataset_dict['train_ulb'])
            self.num_classes = self.num_classes
            
            # Initialize banks on GPU
            self.pslab = torch.zeros(self.num_ulb_samples, self.num_classes).to(self.gpu)
            self.epoch_pslab = torch.zeros(self.num_ulb_samples, self.num_classes).to(self.gpu)
            
            self.print_fn(f"UASD Memory Bank initialized for {self.num_ulb_samples} samples.")

    def train_step(self, x_lb, y_lb, x_ulb_w, idx_ulb, **kwargs):
        # Note: USB passes arguments by name matching. 
        # We MUST add `idx_ulb` to capture the indices of unlabeled data.
        if self.pslab is None:
            # 嘗試從 loader_dict 或 dataset_dict 獲取無標籤數據集大小
            if hasattr(self, 'loader_dict') and 'train_ulb' in self.loader_dict:
                self.num_ulb_samples = len(self.loader_dict['train_ulb'].dataset)
            elif hasattr(self, 'dataset_dict') and 'train_ulb' in self.dataset_dict:
                self.num_ulb_samples = len(self.dataset_dict['train_ulb'])
            else:
                # Fallback: 如果真的找不到，嘗試從參數傳入 (不太建議，但作為最後手段)
                raise ValueError("UASD Error: Cannot determine unlabeled dataset size from loader_dict or dataset_dict.")
            
            self.print_fn(f"Initializing UASD Memory Bank for {self.num_ulb_samples} samples on {self.gpu}...")
            self.pslab = torch.zeros(self.num_ulb_samples, self.num_classes).to(self.gpu)
            self.epoch_pslab = torch.zeros(self.num_ulb_samples, self.num_classes).to(self.gpu)
        
        # 1. Inference on Labeled Data
        logits_x_lb = self.model(x_lb)['logits']
        sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

        # 2. Inference on Unlabeled Data with BN Frozen
        # LAMDA freezes BN statistics during the unlabeled forward pass
        # We simulate this by setting the model to eval mode (for BN) but keeping gradients enabled
        self.model.eval() 
        logits_x_ulb = self.model(x_ulb_w)['logits']
        self.model.train()

        # 3. Memory Bank Logic
        # Retrieve historical average for current batch
        # iter_unlab_pslab = self.pslab[ulb_idx]
        current_pslab = self.pslab[idx_ulb]
        
        # Store current prediction into epoch bank (detached)
        probs_x_ulb = torch.softmax(logits_x_ulb, dim=1)
        with torch.no_grad():
            # self.epoch_pslab[idx_ulb] = probs_x_ulb.detach()
            # [修正] 使用 EMA 即時更新，不依賴 Epoch 結束的全局平均
            # momentum 可以設 0.9 或更低，代表歷史權重
            momentum = 0.9 
            
            # 為了避免初始化為 0 的問題，如果是第一次更新(值為0)，直接覆寫
            is_empty = (self.pslab[idx_ulb].sum(dim=1) == 0)
            
            # 更新非空的部分 (EMA)
            if not is_empty.all():
                self.pslab[idx_ulb[~is_empty]] = momentum * self.pslab[idx_ulb[~is_empty]] + (1 - momentum) * probs_x_ulb[~is_empty].detach()
            
            # 更新空的部分 (直接覆寫)
            if is_empty.any():
                self.pslab[idx_ulb[is_empty]] = probs_x_ulb[is_empty].detach()

        # 4. Target Generation (Self-Distillation)
        # Formula: target = (Historical * (epoch-1) + Current) / epoch
        # Note: self.epoch is 0-indexed in code usually, but UASD logic assumes 1-based counting for averaging
        current_epoch = self.epoch + 1 
        
        # Calculate the soft target used for loss
        # LAMDA: iter_unlab_pslab=(iter_unlab_pslab*(self._epoch-1)+ulb_logits.softmax(1))/self._epoch
        # target_probs = (current_pslab * (current_epoch - 1) + probs_x_ulb.detach()) / current_epoch
        # [修正] 直接使用平滑後的 Memory Bank 作為 Target
        target_probs = self.pslab[idx_ulb]
        
        # 5. Masking
        max_probs, _ = torch.max(target_probs, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        if current_epoch == 5:
            breakpoint()

        # 6. Unsupervised Loss (Cross Entropy with Soft Targets)
        # LAMDA uses CrossEntropyLoss(reduction='none')(ulb_logits, target_probs)
        # Note: PyTorch CE expects class indices for hard labels, or probabilities for soft labels (if supported).
        # USB's self.ce_loss typically handles standard cases. 
        # Here we manually calculate CE with soft targets: -sum(target * log_softmax(input))
        
        log_probs_ulb = torch.log_softmax(logits_x_ulb, dim=1)
        unsup_loss = -(target_probs * log_probs_ulb).sum(dim=1)
        unsup_loss = (unsup_loss * mask).mean()

        # 7. Total Loss with Ramp-up
        # LAMDA: loss=sup_loss+self.lambda_u*(self._epoch/self.epoch)*unsup_loss  <-- Wait, LAMDA logic is (cur/total)? 
        # Actually LAMDA source says: self.lambda_u*(self._epoch/self.epoch) which looks like it might cancel out if names are same?
        # Checking LAMDA source carefully: 
        # loss = sup_loss + self.lambda_u * (self._epoch / self.epoch) * unsup_loss
        # If config.epoch is max_epochs, then it is a linear ramp-up.
        
        ramp_up_ratio = current_epoch / self.epochs # self.epochs is max_epochs in USB
        total_loss = sup_loss + self.lambda_u * ramp_up_ratio * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=None)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def on_train_epoch_end(self):
        # Update the historical memory bank at the end of each epoch
        # LAMDA: self.pslab = ((self._epoch-1)*self.pslab + self.epoch_pslab)/self._epoch
        # current_epoch = self.epoch + 1
        
        # We update the global bank
        # self.pslab = ((current_epoch - 1) * self.pslab + self.epoch_pslab) / current_epoch
        
        # Reset the epoch accumulator for the next epoch? 
        # LAMDA doesn't seem to reset it explicitly to zeros, but overwrites it next time.
        # However, to be safe, we don't strictly need to reset if we overwrite all indices.
        # But since DropLast might be True or dataloader shuffles, resetting is safer practice for debugging,
        # though functionally UASD overwrites `epoch_pslab[idx]` every step.
        pass

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--ulb_loss_ratio', float, 1.0),
        ]