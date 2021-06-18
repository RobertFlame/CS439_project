# CS439_project

# Gradient Compression over SGD and Adam: A Survey

Code for the experimental parts of the course project in CS-439: Optimization for Machine Learning.

The implementation is based on [this repository](https://github.com/epfml/error-feedback-SGD)'s code and uses PyTorch.

## Requirements

The following packages were used for the experiments. Newer versions are also likely to work.

- torchvision==0.2.1
- numpy==1.15.4
- torch==0.4.1
- pandas==0.23.4
- scikit_learn==0.20.3

To install them automatically: `pip install -r requirements.txt`

## Organization

- `optimizers/` contains the custom optimizer, namely CompSGD, ErrorFeedbackSGD and OneBitAdam.
- `models/` contains the deep net architectures. Only Resnet were experimented.
- `results/` contains the results of the experiments in pickle files.
- `utils/` contains utility functions for saving/loading objects, convex optimization, progress bar...
- `checkpoints/` contains the saved models' checkpoints with all the nets parameters. The folder is empty here as those files are very large.

## Notations

We clarify the noations here. In particular,

- **ssgd**: SGD with sign gradient compression.
- **sgd_topk**: SGD with top-*k* gradient compression. 
- **sgd_pcak**: SGD with *k*-PCA gradient compression.
- **sssgd**: SGD with *scaled* sign gradient compression.
- **ussgd**: Unscaled SignSGD (MEM-SGD), i.e., SGD with sign gradient compression and error feedback.
- **ssgdf**: Error-feedback SignSGD, i.e., SGD with *scaled* sign gradient compression and error feedback.
- **onebit_adam_unscaled**: the original version of one-bit Adam.
- **onebit_adam_scaled**: the scaled version of one-bit Adam.

## Usage

- `main.py` can be called from the command line to run a single network training and testing. It can take a variety of optional arguments. Type `python main.py --help` for further details.
- `utils.hyperparameters.py` facilitate the definition of all the hyper-parameters of the experiments.
- `tune_lr.py` allows to tune the learning rate for a network architecture/data set/optimizer configuration.
- `main_experiments.py` contains the experiments in the report. 
- `plot_graph.py` constains the code for plotting the results
- `print_stats.py` constains the code to list the best performance of each experiment done by tunr_lr.py

## Plot

All the figures in the report can be repeated by `run.ipynb`