import math
from matplotlib import pyplot as plt
import torch
import sys
import numpy as np
from glob import glob
from collections import defaultdict


def plot_group_scatter(filename):
    classes_per_group, images_per_class = torch.load(filename)
    classes_list = classes_per_group[0]
    easts = np.array([c[0] for c in classes_list])
    north = np.array([c[1] for c in classes_list])
    easts_centered = easts - easts.mean()
    north_centered = north - north.mean()
    plt.scatter(easts_centered, north_centered, marker='x')
    plt.show()
    """
    print(f'Number of classes: {len(images_per_class)}')
    y = []
    for k in images_per_class:    
        if (k[2] != 0):
            print()
        number_of_classes = len(images_per_class[k])       
        y.append(number_of_classes)
    plt.bar(np.arange(len(y)),y)
    plt.show()
    """

def plot_class_scatter(filename):
    classes_per_group, images_per_class = torch.load(filename)
    i=0
    east_values = []
    north_values = []
    for k in images_per_class:
        filenames = images_per_class[k]
        e_list = []
        n_list = []
        for file in filenames:
            images_metadatas = file.split("@")
            east_value = images_metadatas[1]
            north_value = images_metadatas[2]
            e_list.append(float(east_value))
            n_list.append(float(north_value))
        east_values.append(e_list)
        north_values.append(n_list)
        i += 1
    #for i in range(len(east_values)):
    for i in range(20):
        plt.scatter(east_values[i], north_values[i], marker='x')
    plt.show()


def check_headers():
    dataset_folder = "../small/test"
    images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
    images_metadatas = [p.split("@") for p in images_paths]
    header_set = set()
    for m in images_metadatas:
        header = m[9]
        header_set.add(header) 
    print(header_set)

    images_metadatas = [p.split("@") for p in images_paths]

if __name__ == "__main__":
    plot_class_scatter("./cache/_M10_N5_mipc10.torch")
    