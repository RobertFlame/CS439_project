import pickle 
import numpy as np

def load_obj(name):
    """
    Loads an object from a pickle file.
    :param name: File name.
    :return: The loaded object.
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def find_best(losses_file):
    losses = load_obj(losses_file)
    min_val = np.min(losses)
    min_idx = 0
    for idx, val in enumerate(losses):
        if val == min_val:
            min_idx = idx
    return min_idx

def print_stats(group_name):
    group_name = "{}/".format(group_name)
    config_file = f"{group_name}README.md"
    with open(config_file, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            if line.startswith("lr:"):
                lr = float(line.replace("lr: ", "").replace("\\", ""))
                break
    best_train_idx = find_best(f"{group_name}train_losses")
    best_test_idx = find_best(f"{group_name}test_losses")
    print(f"summary:\n \
        lr: {lr} \n \
        best train idx: {best_train_idx} \n \
        stats: \n \
        train loss: {load_obj(f'{group_name}train_losses')[best_train_idx]}, train acc: {load_obj(f'{group_name}train_accuracies')[best_train_idx]}, \
        test loss: {load_obj(f'{group_name}test_losses')[best_train_idx]}, test acc: {load_obj(f'{group_name}test_accuracies')[best_train_idx]} \n \
        best test idx: {best_test_idx} \n \
        stats: \n \
        train loss: {load_obj(f'{group_name}train_losses')[best_test_idx]}, train acc: {load_obj(f'{group_name}train_accuracies')[best_test_idx]}, \
        test loss: {load_obj(f'{group_name}test_losses')[best_test_idx]}, test acc: {load_obj(f'{group_name}test_accuracies')[best_test_idx]} \n \
    ")

if __name__ == "__main__":
    num = int(input("please input how many lrs you have tested, default is 9: ") or 9)
    for i in range(num):
        print_stats(f"lr{i}")