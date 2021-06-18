"""
Main experiments that were conducted, after the learning rate tuning.
"""

from main import construct_and_train
from utils.hyperparameters import get_experiment_hyperparameters, get_experiment_name
from tune_lr import get_tuned_learning_rate


base_folder = 'main_experiments/'


def run_experiment(model, dataset, optimizer, prefix='', batch_size=128, num_exp=3, start_at=1, gpu=0, lr_val=None):
    base_name = base_folder + 'batchsize-' + str(batch_size) + '/' \
                + prefix + get_experiment_name(model, dataset, optimizer)

    hyperparameters = get_experiment_hyperparameters(model, dataset, optimizer)
    momentum = hyperparameters['momentum']
    weight_decay = hyperparameters['weight_decay']
    comp = hyperparameters['comp']
    noscale = hyperparameters['noscale']
    memory = hyperparameters['memory']
    mnorm = hyperparameters['mnorm']
    mback = hyperparameters['mback']
    exp_asq = hyperparameters['exp_asq']
    adam_or_sgd = hyperparameters['adam_or_sgd']
    start_freeze = hyperparameters['start_freeze']
    k = hyperparameters['k']

    num_epochs = [100, 50, 50]

    for exp_index in range(start_at, num_exp + start_at):
        resume = False
        name = base_name + str(exp_index) + '/'
        lr = get_tuned_learning_rate(model, dataset, optimizer)*batch_size/128 if not lr_val else lr_val
        print('Tuned lr : {}'.format(lr))
        for epochs in num_epochs:
            construct_and_train(name=name, dataset=dataset, model=model, resume=resume, epochs=epochs,
                                lr=lr, batch_size=batch_size, momentum=momentum, weight_decay=weight_decay,
                                comp=comp, k=k, noscale=noscale, memory=memory, mnorm=mnorm, mback=mback, 
                                exp_asq=exp_asq, adam_or_sgd=adam_or_sgd, start_freeze=start_freeze,gpu=gpu, norm_ratio=False)
            resume = True
            lr /= 10


if __name__ == '__main__':
    # run_experiment('vgg', 'cifar10', 'sgdm', batch_size=8)
    # run_experiment('vgg', 'cifar10', 'ssgdf', batch_size=8)
    # run_experiment('vgg', 'cifar10', 'signum', batch_size=8)
    # run_experiment('vgg', 'cifar10', 'sssgd', batch_size=8)

    # run_experiment('vggnonorm', 'cifar10', 'sgdm', batch_size=128)
    # run_experiment('vggnonorm', 'cifar10', 'ssgdf', batch_size=128)
    # run_experiment('vggnonorm', 'cifar10', 'signum', batch_size=128)
    # run_experiment('vggnonorm', 'cifar10', 'sssgd', batch_size=128)
    
    # run_experiment('resnet', 'cifar100', 'sgdm', batch_size=128)
    # run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=128)
    # run_experiment('resnet', 'cifar100', 'signum', batch_size=128)
    # run_experiment('resnet', 'cifar100', 'sssgd', batch_size=128)

    # run_experiment('resnet', 'cifar100', 'sgdm', batch_size=8)
    # run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=8)
    # run_experiment('resnet', 'cifar100', 'signum', batch_size=8)
    # run_experiment('resnet', 'cifar100', 'sssgd', batch_size=8)

    # run_experiment('resnet', 'cifar100', 'sgdm', batch_size=32)
    # run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=32)
    # run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=128)
    # run_experiment('resnet', 'cifar100', 'sssgd', batch_size=32)

    batch_size = 128

    idx = int(input("which one to test? (0-11): "))
    gpu = int(input("Please input which gpu to use: "))

    run_experiment('resnet', 'cifar10', 'sgd', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=0.05623413251903491) if idx == 0 else None
    # run_experiment('resnet', 'cifar10', 'sgdm', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=0.0031622776601683794) if idx == 1 else None
    run_experiment('resnet', 'cifar10', 'sgdm', batch_size=batch_size, num_exp=1, gpu=gpu) if idx == 1 else None
    run_experiment('resnet', 'cifar10', 'signum', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=0.0001) if idx == 2 else None
    run_experiment('resnet', 'cifar10', 'sssgd', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=0.05623413251903491) if idx == 3 else None
    run_experiment('resnet', 'cifar10', 'sgdf', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=2e-3) if idx == 4 else None
    run_experiment('resnet', 'cifar10', 'ssgd', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=0.00005623413251903491) if idx == 5 else None
    run_experiment('resnet', 'cifar10', 'ssgdf', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=0.05623413251903491) if idx == 6 else None
    run_experiment('resnet', 'cifar10', 'sgd_svdk', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=0.01) if idx == 7 else None
    run_experiment('resnet', 'cifar10', 'sgd_topk', batch_size=batch_size, num_exp=1, gpu=gpu, lr_val=0.05623413251903491) if idx == 8 else None
    run_experiment('resnet', 'cifar10', 'adam', batch_size=batch_size, num_exp=1, gpu=gpu, start_at=6) if idx == 9 else None
    run_experiment('resnet', 'cifar10', 'onebit_adam_unscaled', batch_size=batch_size, num_exp=1, gpu=gpu, start_at=6) if idx == 10 else None
    run_experiment('resnet', 'cifar10', 'onebit_adam_scaled', batch_size=batch_size, num_exp=1, gpu=gpu, start_at=6) if idx == 11 else None