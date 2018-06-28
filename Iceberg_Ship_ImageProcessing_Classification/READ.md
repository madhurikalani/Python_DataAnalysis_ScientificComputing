# Image Processing_Iceberg_Ship

## Simply run the file  "Image_Processing_Iceberg_Ship.py"

## Data is stored in data/
train_data: train.json

## Need to install the following package if not present

## Data Manipulation
import pandas as pd
pd.options.display.max_columns = 25
import numpy as np
from IPython.display import display

## Visualizations
import matplotlib.pyplot as plt
%matplotlib inline
import pylab as plb
import seaborn as sns
sns.set_style("whitegrid", {"axes.grid": False})
import missingno as msno
import cv2
from scipy import signal
