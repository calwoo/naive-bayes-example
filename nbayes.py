import numpy as np
import pandas as pd 
from utils import *

"""
The dataset we are using the Pima indians diabetes dataset from UC Irvine. It comprises medical data from 768
patients and describes various measurements. The final column indicates whether the patient has diabetes or not.
"""
data = pd.read_csv("pima-indians-diabetes.data.csv", header=None)
# convert to numpy
data = data.values
train, test = split_data(data)
print(len(train), len(test), len(data))
