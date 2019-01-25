import numpy as np 


def split_data(data, split_ratio=0.67):
    size = len(data)
    train_l = int(size * split_ratio)
    test_l = size - train_l
    # split into train and test
    indices = np.random.choice(size, train_l, replace=False)
    train = [data[i] for i in indices]
    test = [data[i] for i in range(size) if i not in indices]
    return train, test

def gen_labels(data):
    data = np.array(data)
    size = data.shape[1] - 1
    labels = data[:,-1]
    data = data[:,:size]
    return data, labels