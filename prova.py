import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test
import util
import my_parser as parser
import commons
import cosface_loss
import sphereface_loss
import arcface_loss
import augmentations
from model import network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.target_dataset import TargetDataset
from torch.utils.data import DataLoader
from itertools import chain

args = parser.parse_arguments()
folder = "tokyo_xs/test/night/"


dataset = TrainDataset(args, folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                        current_group=0, min_images_per_class=args.min_images_per_class) 

loader = commons.InfiniteDataLoader(0, num_workers=1,#args.num_workers,
                                                batch_size=32, shuffle=True,
                                                 drop_last=True)

print(next(loader))