import numpy as np
import torch
import math
from torch.optim.optimizer import Optimizer, required


def scaled_sign(x):
    """
    :param x: torch Tensor
    :return: The sign tensor scaled by it's L1 norm and divided by the number of elements
    """
    return x.norm(p=1) / x.nelement() * torch.sign(x)


def unscaled_sign(x):
    """
    This is the standard sign compression. It has been experimented to give worse test accuracies than the scaled
    counter part.
    :param x: torch Tensor
    :return: sign(tensor)
    """
    return torch.sign(x)


class OneBitAdam(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, comp='scaled_sign', memory=False, amsgrad=False, start_freeze=20):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if comp == 'scaled_sign':
            comp = scaled_sign
        elif comp == 'sign':
            comp = unscaled_sign
        elif not callable(comp) and comp is not None:
            raise ValueError("Invalid comp value: {} (must be callable or None)".format(comp))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, freeze_step=391*start_freeze,
                        comp=comp, memory=memory, amsgrad=amsgrad)

        super(OneBitAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['memory'] = torch.zeros_like(p.data)

                # To compute the gradients norms ratios over time
                param_state['dim'] = p.nelement()
                param_state['gradient'] = None
                param_state['corrected_gradient'] = None

    def __setstate__(self, state):
        super(OneBitAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            eps = group['eps']
            comp = group['comp']
            memory = group['memory']

            for p in group['params']:
                param_state = self.state[p]
                if p.grad is None:
                    continue
                
                if 'freeze_sq' not in param_state:
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    if 'exp_avg' not in param_state:
                        param_state['step'] = 0
                        param_state['exp_avg'] = torch.zeros_like(p.data)
                        param_state['exp_asq'] = torch.zeros_like(p.data)
                    param_state['step'] += 1
                    param_state['exp_avg'].mul_(group['betas'][0]).add_(d_p, alpha=1 - group['betas'][0])
                    param_state['exp_asq'].mul_(group['betas'][1]).addcmul_(d_p, d_p, value=1 - group['betas'][1])

#                     bias_correction1 = 1 - group['betas'][0] ** param_state['step']
#                     bias_correction2 = 1 - group['betas'][1] ** param_state['step']
                    bias_correction1 = 1.0
                    bias_correction2 = 1.0
                    denom = (param_state['exp_asq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    d_p = param_state['exp_avg'] / bias_correction1

                    param_state['gradient'] = d_p  # Save the gradient so its norm can be computed later
                    param_state['corrected_gradient'] = d_p

                    d_p = group['lr'] * d_p / denom

                    p.data.add_(-1, d_p)
        
                    if param_state['step'] >= group['freeze_step']:
                        print('OnebitAdam - starting compression')
                        param_state['freeze_sq'] = True

                else:  # frozen and using compression
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    if 'exp_avg' not in param_state:
                        param_state['step'] = 0
                        param_state['exp_avg'] = torch.zeros_like(p.data)
                        param_state['exp_asq'] = torch.zeros_like(p.data)
                    param_state['step'] += 1
                    param_state['exp_avg'].mul_(group['betas'][0]).add_(d_p, alpha=1 - group['betas'][0])
#                     param_state['exp_asq'].mul_(group['betas'][1]).addcmul_(d_p, d_p, value=1 - group['betas'][1])  # freeze

                    bias_correction1 = 1 - group['betas'][0] ** param_state['step']
                    bias_correction2 = 1 - group['betas'][1] ** param_state['step']
                    denom = (param_state['exp_asq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    d_p = param_state['exp_avg'] / bias_correction1

                    # d_p corresponds to g in alg. 1 from the paper.
                    param_state['gradient'] = d_p  # Save the gradient so its norm can be computed later

                    d_p = group['lr'] * d_p
                    corrected_gradient = param_state['memory'] + d_p

                    # Save the corrected gradient to compute the norms
                    param_state['corrected_gradient'] = corrected_gradient

                    if comp is not None:
                        corrected_gradient = comp(corrected_gradient)

                    ''' hack to scale the signed gradient by the learning
                        rate since torch.sign(x) ignores the learning rate '''
                    if comp == unscaled_sign:
                        corrected_gradient = group['lr'] * corrected_gradient

                    if memory:
                        param_state['memory'] = param_state['memory'] + d_p - corrected_gradient

                    p.data.add_(-1, corrected_gradient / denom)

    
        return loss

    def memory_norm(self):
        """
        :return: The L2 norm of the memory (if any)
        """
        norm = 0
        for group in self.param_groups:
            for p in group['params']:
                n = p.norm()
                norm += float(n * n)

        return np.sqrt(norm)
    
    def exp_asq(self):
        """
        :return: The list of tensors of exp_asq
        """
        ret = []

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                ret.append(param_state['exp_asq'].cpu())
        
        return ret

    def gradient_norms_ratio(self):
        res = []
        sum_l2_norms = 0
        sum_normalized_l1_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                n1 = param_state['gradient'].norm(p=1)
                n2 = param_state['gradient'].norm(p=2)
                d = param_state['dim']
                sum_l2_norms += n2*n2
                sum_normalized_l1_norm += n1*n1/d
                res.append(n1*n1/n2/n2/d)
        ''' Correct ratio = (sum of (n1)^2/d)/(sum of (n2)^2).
            The last coordinate of res has the correct ratio. '''
        res.append(sum_normalized_l1_norm/sum_l2_norms)
    
        return np.array(res)

    def corrected_gradient_norms_ratio(self):
        res = []
        sum_l2_norms = 0
        sum_normalized_l1_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                n1 = param_state['corrected_gradient'].norm(p=1)
                n2 = param_state['corrected_gradient'].norm(p=2)
                d = param_state['dim']
                sum_l2_norms += n2*n2
                sum_normalized_l1_norm += n1*n1/d
                res.append(n1*n1/n2/n2/d)
        ''' Correct ratio = (sum of (n1)^2/d)/(sum of (n2)^2).
            The last coordinate of res has the correct ratio. '''
        res.append(sum_normalized_l1_norm/sum_l2_norms)
        return np.array(res)

    def params_dims(self):
        res = []
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                d = param_state['dim']
                res.append(d)
        return np.array(res)
