import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import time
import argparse
import shutil

num_classes = 43

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, help="directory of data folders")
opts = parser.parse_args()

root_test_input = opts.root + "/"
root_test_output = opts.root + "/Test_Arranged/"
root_test_csv = opts.root + "/Test.csv"

if not os.path.isdir('Test_Arranged'):
    os.mkdir('Test_Arranged')

for i in range(num_classes):
    if not os.path.isdir('Test_Arranged/' + str(i)):
        os.mkdir('Test_Arranged/' + str(i))


dTest = pd.read_csv(root_test_csv)
dTestPath = root_test_input + dTest["Path"]
dTestClass = dTest["ClassId"]

for i in range(len(dTestClass)):
    shutil.copy(dTestPath[i], root_test_output + str(dTestClass[i]))
