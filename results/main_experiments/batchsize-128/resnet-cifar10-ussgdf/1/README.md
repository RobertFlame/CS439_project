name: main_experiments/batchsize-128/resnet-cifar10-ussgdf/1/\
dataset: cifar10\
model: resnet\
resume: False\
epochs: 100\
lr: 0.002\
batch_size: 128\
momentum: 0\
weight_decay: 0.0005\
comp: sign\
k: 0\
noscale: False\
memory: True\
mnorm: True\
mback: True\
norm_ratio: False\
exp_asq: False\
adam_or_sgd: sgd\
start_freeze: 20\

name: main_experiments/batchsize-128/resnet-cifar10-ussgdf/1/\
dataset: cifar10\
model: resnet\
resume: True\
epochs: 50\
lr: 0.0002\
batch_size: 128\
momentum: 0\
weight_decay: 0.0005\
comp: sign\
k: 0\
noscale: False\
memory: True\
mnorm: True\
mback: True\
norm_ratio: False\
exp_asq: False\
adam_or_sgd: sgd\
start_freeze: 20\

name: main_experiments/batchsize-128/resnet-cifar10-ussgdf/1/\
dataset: cifar10\
model: resnet\
resume: True\
epochs: 50\
lr: 2e-05\
batch_size: 128\
momentum: 0\
weight_decay: 0.0005\
comp: sign\
k: 0\
noscale: False\
memory: True\
mnorm: True\
mback: True\
norm_ratio: False\
exp_asq: False\
adam_or_sgd: sgd\
start_freeze: 20\
