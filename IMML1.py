import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/ameliakratzer/Desktop/LinInterpolation/ML/COO.csv')
