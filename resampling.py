# resampling - used to estimate the performance of a model on new data

"""
train and test split

- split dataset into training and test datasets
- training dataset is used to train a model
- test dataset is used to measure model performance 

When comparing algorithms, use the same train/test datasets and seed 
the random number generator the same way before splitting the data.
"""
from random import seed
from random import randrange 

"""
split into train and test datasets. 

- arguments: dataset, split percentage
- the dataset is split as a list of lists
- default split percentage is 60%. training data 60%, test data 40%. 
"""
def train_test_split(dataset, split=0.60): 
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    # random rows are removed from dataset copy 
    # and added to training dataset until train size (%) reached
    while len(train) < train_size:
        # randrange() returns a random integer in range 0 and list size
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    # return training dataset and dataset_copy (remaining data comprising test dataset)
    return train, dataset_copy

# example train/test split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
train, test = train_test_split(dataset)
print(train)
print(test) 
"""
[[3], [2], [7], [1], [8], [9]]
[[4], [5], [6], [10]]
"""

"""
k-fold Cross-Validation Split

- provides more accurate performance estimate than train/test
- data is split into k groups
- each group is a 'fold'
- the kth group is the test dataset

1) algorithm is trained 
2) algorithm is evaluated k times 
3) algorithm performance summarized by mean performance score
"""

