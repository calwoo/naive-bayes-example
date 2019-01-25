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
# split to data and labels
train_data, train_labels = gen_labels(train)
test_data, test_labels = gen_labels(test)

"""
Now we set up our Gaussian priors for the data. For each classification value y=0 or 1, we will compute the mean
and variance of each feature and use these as the parameters for our Gaussians. This is equivalent of finding MLE
estimates under the given independence relations that naive bayes imposes.
"""
def separate_classes(data, labels, num_classes=2):
    separated = {}
    for K in range(num_classes):
        separated[K] = []
        for i in range(data.shape[0]):
            if int(labels[i]) == K:
                separated[K].append(data[i])
    return separated

def find_parameters(data):
    parameters = []
    data = np.array(data)
    num_features = data.shape[1]
    for feature in range(num_features):
        col = data[:, feature]
        mean = np.mean(col)
        stddev = np.std(col)
        parameters.append((mean, stddev))
    return parameters

# put it all together
def parameters(data, labels, num_classes=2):
    separated = separate_classes(data, labels, num_classes)
    params = {}
    for K in range(num_classes):
        pa = find_parameters(separated[K])
        params[K] = pa
    return params

params = parameters(train_data, train_labels)

"""
Now we have the parameters to our Gaussian priors. We can then use these to compute probabilities following Bayes
rule to get the posteriors. Remember, we want to compute p(y|x), the condition of the latent variables given the
feature variables.
"""
def gaussian(x, mean, stddev):
    gauss = np.exp(-(x - mean)**2 / (2 * stddev**2))
    return (1 / np.sqrt(2*np.pi*stddev**2)) * gauss

def probabilities(datapoint, parameters):
    probs = {}
    


