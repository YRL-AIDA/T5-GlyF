import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Levenshtein import ratio

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def load_pkl(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def save_pkl(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def parse_args(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config_path', type=str, required=True, help='Input path to the configuration file')
    return parser.parse_args()

def calculate_accruracy(y_pred, y_true):
    accuracy = 0
    test_size = len(y_pred)
    for i in range(test_size):
        if y_pred[i] == y_true[i]:
            accuracy += 1
    return accuracy/test_size

def calculate_levenshtein_ratio(y_pred, y_true):
    l_ratio = []
    test_size = len(y_pred)
    for i in range(test_size):
        l_ratio.append(ratio(y_true[i], y_pred[i]))
    return np.mean(l_ratio)

def plot_losses(train_loss, val_loss, save_chart_path='losses.png'):
    c_iter = range(len(train_loss))

    fig = plt.figure(figsize=(15, 7), dpi=80)
    ax = fig.gca()
    ax.set_xticks(c_iter)
    plt.plot(c_iter, train_loss, color='orange', label='train_loss')
    plt.plot(c_iter, val_loss, color='blue', label='val_loss')

    plt.legend(loc='upper right')
    plt.title('Losses')
    plt.grid()
    plt.savefig(save_chart_path)