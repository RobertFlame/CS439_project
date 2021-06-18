import numpy as np
import numpy.linalg as LA
import torch
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

def _resize_to_2d(x):
    """
    x.shape > 2
    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    shape = x.shape
    if x.ndim == 1:
        n = x.shape[0]
        return x.reshape((n//2, 2))
    if all([s == 1 for s in shape[2:]]):
        return x.reshape((shape[0], shape[1]))
    # each of (a, b) has related features
    x = x.reshape((shape[0], shape[1], -1))
    # stack those related features into a tall matrix
    x_tmp = x.reshape((shape[0]*shape[1], -1))
    tmp_shape = x_tmp.shape
    return x_tmp.reshape((int(tmp_shape[0]/2), int(tmp_shape[1]*2)))

class svdCompress:
    def __init__(self, k):
        self.k = k

    def __call__(self, grad):
        k = self.k
        device = grad.device
        grad = grad.cpu().numpy()
        orig_size = list(grad.shape)
        ndims = grad.ndim
        
        if ndims != 2:
            grad = _resize_to_2d(grad)
            shape = list(grad.shape)
            ndims = len(shape)

        # encode
        u, s, vT = LA.svd(grad, full_matrices=False)     
        u = u[:, :k]
        s = s[:k]
        #  v = v[:, :self.svd_rank]
        vT = vT[:k, :]
            
        # decode
        grad = np.dot(np.dot(u, np.diag(s)), vT)
        grad = torch.Tensor(grad)
        grad = grad.view(orig_size)

        return grad.to(device)


class topkCompress:
    def __init__(self, k):
        self.k = k

    def __call__(self, grad):
        device = grad.device
        grad = grad.cpu().numpy()
        k = self.k

        indices = np.argpartition(np.abs(grad.ravel()), -k)[-k:]
        out_grad = np.zeros_like(grad).ravel()
        out_grad[indices] = grad.ravel()[indices]
        out_grad = torch.Tensor(grad).reshape(grad.shape)

        return out_grad.to(device)

class randkCompress:
    def __init__(self, k): 
        self.k = k

    def __call__(self, grad):
        device = grad.device
        grad = grad.cpu().numpy()
        k = self.k

        d = np.prod(grad.shape)
        indices = np.random.choice(d, k, replace=False)
        out_grad = np.zeros_like(grad).ravel()
        out_grad[indices] = grad.ravel()[indices]
        out_grad = torch.Tensor(out_grad).reshape(grad.shape)

        return out_grad.to(device)

class CompSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, comp='scaled_sign', k=1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if comp == 'scaled_sign':
            comp = scaled_sign
        elif comp == 'sign':
            comp = unscaled_sign
        elif comp == 'svdk':
            comp = svdCompress(k=k)
        elif comp == 'topk':
            comp = topkCompress(k=k)
        elif comp == 'randk':
            comp = randkCompress(k=k)
        elif not callable(comp) and comp is not None:
            raise ValueError("Invalid comp value: {} (must be callable or None)".format(comp))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        comp=comp, k=k)
        self.k = k

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CompSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                # To compute the gradients norms ratios over time
                param_state['dim'] = p.nelement()
                param_state['gradient'] = None

    def __setstate__(self, state):
        super(CompSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            comp = group['comp']

            for p in group['params']:
                param_state = self.state[p]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # d_p corresponds to g in alg. 1 from the paper.
                param_state['gradient'] = d_p  # Save the gradient so its norm can be computed later

                # compress the gradient
                compressed_gradient = comp(d_p)
                
                p.data.add_(-group['lr'], compressed_gradient)

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
        return self.gradient_norms_ratio()

    def params_dims(self):
        res = []
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                d = param_state['dim']
                res.append(d)
        return np.array(res)