import torch
import torch.nn as nn
import numpy as np
import random

class PruneOp():
    def __init__(self, model, target_sparsity):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1
        assert(count_targets == len(target_sparsity))
        self.target_sparsity = target_sparsity
        self.count_targets = count_targets
        self.masks = []
        self.target_modules = []
        self.num_params = 0
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in range(self.count_targets):
                    self.masks.append(m.weight.new_ones(m.weight.size(), dtype=torch.uint8))
                    self.target_modules.append(m.weight)
                    self.num_params += m.weight.numel()

    # setup the mask for pruning
    # no actual pruning is conducted
    def init_pruning(self):
        self.update_masks(2.0, 1.0)
                    
    def get_sparsity(self):
        num_params_active = 0
        for mask in self.masks:
            num_params_active += mask.sum()
        print('LOG_get_sparsity: ', self.num_params-num_params_active.item(), 'of', self.num_params)
        return 1.0 - num_params_active.item() / self.num_params

    def get_masks(self):
        return self.masks

    def set_masks(self, masks):
        for index in range(self.count_targets):
            self.masks[index].copy_(masks[index])

    def mask_params(self):
        for index in range(self.count_targets):
            self.target_modules[index].data[1-self.masks[index]] = 0

    def mask_grad(self):
        for index in range(self.count_targets):
            self.target_modules[index].grad.data[1-self.masks[index]] = 0 
        
    def update_masks(self, update_p, alpha):
        if update_p is None:
            return
        for index in range(self.count_targets):
            if random.random() < update_p: # update mask
                w_np = np.abs(self.target_modules[index].data.cpu().numpy())
                sorted_w_np = np.sort(w_np.reshape(-1))
                sparsity_index = np.round(self.target_sparsity[index] * alpha * w_np.size).astype(np.int)
                sparsity_thresh = sorted_w_np[sparsity_index]
                self.masks[index] = self.target_modules[index].data.abs()>sparsity_thresh

