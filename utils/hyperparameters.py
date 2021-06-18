"""
Makes the hyper-parameters selection easier for each experiment.
"""


def get_experiment_hyperparameters(model, dataset, optimizer):
    """
    :param model: 'vgg', 'vggnonorm', 'resnet' or 'lstm'
    :param dataset: 'cifar10' or 'cifar100'
    :param optimizer: 'sgdm', 'ssgd' or 'sssgd'
    :return: A dictionary with the hyper-parameters
    """
    hyperparameters = dict()
    if model != 'vgg' and model != 'vggnonorm' and model != 'resnet' and model != 'lstm':
        raise ValueError('Invalid value for model : {}'.format(model))
    if dataset != 'cifar10' and dataset != 'cifar100':
        raise ValueError('Invalid value for dataset : {}'.format(dataset))
    momentum = 0
    comp = ''
    noscale = False
    memory = False
    mback = False
    mnorm = False
    exp_asq = False
    adam_or_sgd = 'sgd'
    start_freeze = 20
    weight_decay = 5e-4
    k = 0
    if optimizer == 'sgd':
        comp = 'sgd'
    elif optimizer == 'sgdm':
        momentum = 0.9
        comp = 'sign'
    elif optimizer == 'ssgd':
        noscale = True
        comp = 'sign'
    elif optimizer == 'sssgd':
        comp = 'scaled_sign'
    elif optimizer == 'ssgdf':
        memory = True
        mback = True
        mnorm = True
        comp = 'scaled_sign'
    elif optimizer == 'ussgdf':
        memory = True
        mback = True
        mnorm = True
        comp = 'sign'
    elif optimizer == 'signum':
        noscale = True
        comp = 'sign'
        momentum = 0.9
    elif optimizer == 'sgd_pcak':
        comp = 'svdk'
        k = 3
    elif optimizer == 'sgd_topk':
        comp = 'topk'
        k = 10
    elif optimizer == 'adam':
        adam_or_sgd = 'adam'
        comp = 'sign'
        exp_asq = True
        start_freeze = 1e7
        weight_decay = 0
    elif optimizer == 'onebit_adam_unscaled':
        adam_or_sgd = 'adam'
        comp = 'sign'
        exp_asq = True
        weight_decay = 0
    elif optimizer == 'onebit_adam_scaled':
        adam_or_sgd = 'adam'
        comp = 'scaled_sign'
        exp_asq = True
        weight_decay = 0
    else:
        raise ValueError('Invalid value for optimizer : {}'.format(optimizer))

    hyperparameters['momentum'] = momentum
    hyperparameters['comp'] = comp
    hyperparameters['noscale'] = noscale
    hyperparameters['memory'] = memory
    hyperparameters['mback'] = mback
    hyperparameters['mnorm'] = mnorm
    hyperparameters['weight_decay'] = weight_decay
    hyperparameters['exp_asq'] = exp_asq
    hyperparameters['adam_or_sgd'] = adam_or_sgd
    hyperparameters['start_freeze'] = start_freeze
    hyperparameters['k'] = k

    return hyperparameters


def get_experiment_name(model, dataset, optimizer):
    """
    Name where the experiment's results are saved
    :param model: 'vgg', 'vggnonorm', 'resnet' or 'lstm'
    :param dataset: 'cifar10' or 'cifar100'
    :param optimizer: 'sgdm', 'ssgd' or 'sssgd'
    :return: The name of the experiment
    """
    return model + '-' + dataset + '-' + optimizer + '/'
