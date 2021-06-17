import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import pickle


def plot_lr_curve(lr_curve_group, fig_name, align_axis=True, yrange=None):
    '''
    plot function
    params:
        lr_curve_group: a list of curve_group constructed by function generate_curve_group
        fig_name: the output figure name
        align_axis: if you have more than one curve_group in lr_curve_group, set to True to make their ylim the same
        yrange: use it if you want to manually set the ylim of the graph. This variable has high priority than align_axis
    '''
    num_subfigures = len(lr_curve_group)

    fig, axs = plt.subplots(1, num_subfigures, figsize=(6*num_subfigures-1,5))

    colors = ['b', 'g', 'c', 'm', 'k', 'r']
    lines = ["-", "--", "-.", ":"]

    for count, curves in enumerate(lr_curve_group):
        y_label = curves.get("ylabel", "")
        legend_loc = curves.get("loc", 4)
        ax = axs if len(lr_curve_group) == 1 else axs[count]
        for count2, [curve, name, early_stop] in enumerate(curves["curves"]):
            ax.plot([p[0] for p in curve], [p[1] for p in curve], color=colors[count2 % len(colors)], linestyle=lines[count2 % len(lines)], label=name)
            if early_stop:
                ax.scatter(curve[early_stop][0], curve[early_stop][1], c="red", s=10)
        if yrange:
            ax.set_ylim(yrange[0], yrange[1])
        elif align_axis:
            min_bound = np.inf
            max_bound = -np.inf
            for curves in lr_curve_group:
                ys = []
                for curve_settings in curves["curves"]:
                    ys += [p[1] for p in curve_settings[0]]
                min_bound = min(min_bound, np.min(ys))
                max_bound = max(max_bound, np.max(ys))

            ax.set_ylim(0.99*min_bound, 1.01*max_bound)

        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel("Iterations", fontsize=14)
        if y_label != "":
            ax.set_ylabel(y_label, fontsize=14)
        ax.legend(loc=legend_loc, prop={'size': 14})

    plt.savefig(fig_name, dpi=600)


def load_obj(name):
    """
    Loads an object from a pickle file.
    :param name: File name.
    :return: The loaded object.
    """
    with open(name + '.pkl', 'rb') as f:
        values = pickle.load(f)
        return [[idx, val] for idx, val in enumerate(values)]


def find_early_stop(curve):
    # use it if you want to show in the graph where is the early-stop point. The input is test loss
    min_loss = np.inf
    min_idx = -1
    for ite, loss in curve:
        if min_loss > loss:
            min_loss = loss
            min_idx = ite
    return min_idx


def generate_curve_group(group_label, logs_file, exps, use_early_stop):
    '''
    generate a set of information for plotting
    params:
        group_label: the y label used for plotting
        logs_file: the filename of the result you want to use
        exps: a list of which experiments you want to include in this graph
        use_early_stop: set to True if you want to show the iteration with minimum test loss
    '''
    prefix="results/main_experiments/batchsize-128/resnet-cifar10-"
    logs = {exp: load_obj(prefix+exp+"/1/"+logs_file) for exp in exps}
    if use_early_stop:
        test_losses = {exp: load_obj(prefix+exp+"/1/test_losses") for exp in exps}
        early_stop = {exp: find_early_stop(test_losses[exp]) for exp in exps}
    return {
        "ylabel": group_label,
        "curves": [
            [logs[exp], exp, early_stop[exp] if use_early_stop else None] for exp in exps
        ]
    }


def exp(group_logs, exps, fig_name, yrange=None, use_early_stop=False):
    '''
    plot the results given the set of values, its labels, corresponding files, and the output figure name
    params:
        group_logs: a dict storing pairs of <ylabel of the graph, the filename of this result>
        exps: a list of which experiments you want to include in this graph
        fig_name: the output figure name
        yrange: use it if you want to manually set the ylim of the graph
        use_early_stop: set to True if you want to show the iteration with minimum test loss
    '''
    curve_groups =  [
        generate_curve_group(group_label, logs_file, exps, use_early_stop) for group_label, logs_file in group_logs.items()
    ]

    plot_lr_curve(curve_groups, fig_name+".png", yrange=yrange)


if __name__ == "__main__":
    exps_all = ["sgd", "ssgdf", "sssgd", "ssgd", "signum", "sgdm"]
    exps =  ["sgd", "ssgdf", "sssgd", "ssgd", "signum"]

    example_group_logs = {
        "train accuracy":"train_accuracies",
        "test accuracy":"test_accuracies",
        "train loss":"train_losses",
        "test_loss":"test_losses",
        "memory norm":"memory_norms"
    }

    exp({"train accuracy":"train_accuracies"}, exps, "all_train_accuracy_wo_sgdm")
    exp({"train accuracy":"train_accuracies"}, exps, "zoom_all_train_accuracy_wo_sgdm", yrange=[94.0, 100.0])
    exp({"train loss":"train_losses"}, exps, "all_train_loss_wo_sgdm")
    exp({"test accuracy":"test_accuracies"}, exps, "all_test_accuracy_wo_sgdm")
    exp({"test accuracy":"test_accuracies"}, exps, "zoom_all_test_accuracy_wo_sgdm", yrange=[75.0, 95.0])
    exp({"test loss":"test_losses"}, exps, "all_test_loss_wo_sgdm")
    exp({"memory norm":"memory_norms"}, ["ssgdf"], "ssgdf_memory_norm")
