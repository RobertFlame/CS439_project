"""
Script used to tune the learning rates of the models. It was tuned only for a batch size of 128 and
was then scaled appropriately for other batch sizes.
"""

import numpy as np

from main import construct_and_train
from utils.pickle import save_obj, load_obj
from utils.hyperparameters import get_experiment_hyperparameters, get_experiment_name


base_folder = 'lr_tuning/'
num_epochs = 100
batch_size = 128


def get_tuned_learning_rate(model, dataset, optimizer):
    """
    Returns the learning rate for the given experiment once the tuning has been made.
    :param model: 'vgg', 'vggnonorm', 'resnet' or 'lstm'
    :param dataset: 'cifar10' or 'cifar100'
    :param optimizer: 'sgdm', 'ssgd' or 'sssgd'
    :return: lr
    """
    name = base_folder + get_experiment_name(model, dataset, optimizer)
    lr_space = load_obj('./results/' + name + 'lr_space')
    losses = load_obj('./results/' + name + 'losses')
    return lr_space[np.nanargmin(losses)]


def tune_learning_rate(model, dataset, optimizer, base_name=None, gpu=0):
    """
    Tune the learning rate for a given experiment (batch size 128)
    The results are saved in base_folder + experiment_name if base_name is None,
    or in base_folder + base_name otherwise
    :param model: 'vgg', 'vggnonorm', 'resnet' or 'lstm'
    :param dataset: 'cifar10' or 'cifar100'
    :param optimizer: 'sgdm', 'ssgd' or 'sssgd'
    :param base_name: If you want to have a custom name for the saving folder
    """
    model = model.lower()
    dataset = dataset.lower()
    optimizer = optimizer.lower()

    if base_name is None:
        base_name = base_folder + get_experiment_name(model, dataset, optimizer)
    else:
        base_name = base_folder + base_name

    hyperparameters = get_experiment_hyperparameters(model, dataset, optimizer)
    momentum = hyperparameters['momentum']
    weight_decay = hyperparameters['weight_decay']
    comp = hyperparameters['comp']
    noscale = hyperparameters['noscale']
    memory = hyperparameters['memory']
    mnorm = hyperparameters['mnorm']
    mback = hyperparameters['mback']
    k = hyperparameters['k']
    exp_asq = hyperparameters['exp_asq']
    adam_or_sgd = hyperparameters['adam_or_sgd']
    start_freeze = hyperparameters['start_freeze']

    losses = []
    # lr_space = np.logspace(-5, 1, 9)
    # lr_space = np.logspace(-7, -1, 9)
    for index, lr in enumerate(lr_space):
        name = base_name + 'lr' + str(index)
        res = construct_and_train(name=name, dataset=dataset, model=model, resume=False, epochs=num_epochs,
                                  lr=lr, batch_size=batch_size, momentum=momentum, weight_decay=weight_decay,
                                  comp=comp, k=k, noscale=noscale, memory=memory, mnorm=mnorm, mback=mback, 
                                  exp_asq=exp_asq, adam_or_sgd=adam_or_sgd, start_freeze=start_freeze, gpu=gpu, norm_ratio=False)
        best_loss = np.nanmin(res['test_losses'])
        losses.append(best_loss)
    losses = np.array(losses)
    save_obj(lr_space, './results/' + base_name + 'lr_space')
    save_obj(losses, './results/' + base_name + 'losses')
    with open('./results/' + base_name + 'README.md', 'w') as file:
        file.write('Best learning rate : {}\\\n'.format(lr_space[np.nanargmin(losses)]))
        file.write('Best loss reached over {0} epochs : {1}\n'.format(num_epochs, np.nanmin(losses)))


if __name__ == '__main__':
    """
    tune_learning_rate('vgg', 'cifar10', 'sgdm')
    tune_learning_rate('vgg', 'cifar10', 'ssgd')
    tune_learning_rate('vgg', 'cifar10', 'sssgd')
    tune_learning_rate('vgg', 'cifar10', 'ssgdf')
    """

    """
    tune_learning_rate('resnet', 'cifar100', 'sgdm')
    tune_learning_rate('resnet', 'cifar100', 'ssgd')
    tune_learning_rate('resnet', 'cifar100', 'sssgd')
    tune_learning_rate('resnet', 'cifar100', 'ssgdf')
    """

    """
    tune_learning_rate('vgg', 'cifar10', 'signum')
    tune_learning_rate('resnet', 'cifar100', 'signum')
    """
    idx = int(input("which one to test? (0-11): "))
    gpu = int(input("Please input which gpu to use: "))
    
    tune_learning_rate('resnet', 'cifar10', 'sgd', gpu=gpu) if idx == 0 else None
    tune_learning_rate('resnet', 'cifar10', 'sgdm', gpu=gpu) if idx == 1 else None
    tune_learning_rate('resnet', 'cifar10', 'signum', gpu=gpu) if idx == 2 else None
    tune_learning_rate('resnet', 'cifar10', 'sssgd', gpu=gpu) if idx == 3 else None
    tune_learning_rate('resnet', 'cifar10', 'sgdf', gpu=gpu) if idx == 4 else None
    tune_learning_rate('resnet', 'cifar10', 'ssgdf', gpu=gpu)if idx == 5 else None
    tune_learning_rate('resnet', 'cifar10', 'sgd_pcak', gpu=gpu) if idx == 6 else None
    tune_learning_rate('resnet', 'cifar10', 'sgd_topk', gpu=gpu) if idx == 7 else None
    tune_learning_rate('resnet', 'cifar10', 'ssgd', gpu=gpu) if idx == 8 else None
    tune_learning_rate('resnet', 'cifar10', 'adam', gpu=gpu) if idx == 9 else None
    tune_learning_rate('resnet', 'cifar10', 'onebit_adam_unscaled', gpu=gpu) if idx == 10 else None
    tune_learning_rate('resnet', 'cifar10', 'onebit_adam_scaled', gpu=gpu) if idx == 11 else None

    # Sign, noscale, no memory, momentum 0.9, weight decay
