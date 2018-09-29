import numpy as np 
import pandas as pd
import matplotlib.pylot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split

data=pd.read_csv('mnist.csv')
