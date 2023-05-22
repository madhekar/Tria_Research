#Import suppporting libraries
import tarfile
import urllib.request as urllib2
import os
from os import listdir
from os.path import isfile, join
import re
#Import deep learning libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, models
import torchvision.models as models
#Import data analytics libraries
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
#Import image visualization libraries
from PIL import *
from PIL import ImageFile
from PIL import Image
#System settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['WANDB_CONSOLE'] = 'off'
#Coloring for print outputs
class color:
   RED = '\033[91m'
   BOLD = '\033[1m'
   END = '\033[0m'


def getting_data(url,path):
  data = urllib2.urlopen(url)
  tar_package = tarfile.open(fileobj=data, mode='r:gz')
  tar_package.extractall(path)
  tar_package.close()
  return print("Data extracted and saved.")

getting_data("http://ai.stanford.edu/~jkrause/car196/car_ims.tgz","/content/carimages")

def getting_metadata(url,filename):
  '''
  Downloading a metadata file from a specific url and save it to the disc.
  '''
  labels = urllib2.urlopen(url)
  file = open(filename, 'wb')
  file.write(labels.read())
  file.close()
  return print("Metadata downloaded and saved.")

getting_metadata("http://ai.stanford.edu/~jkrause/car196/cars_annos.mat","car_metadata.mat")
