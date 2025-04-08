# Homework 12
# Problem 2

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# Problem 1 (a)

data = pd.read_csv(r"C:\\Users\\npb28.la\\8250_Computation\\HW12\\dataset(1).csv")
dataset = data.to_numpy()

data_sep = pd.read_csv(r"C:\\Users\\npb28.la\\8250_Computation\\HW12\\dataset_separable.csv")
dataset_sep = data_sep.to_numpy()