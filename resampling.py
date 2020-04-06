# resampling - used to estimate the performance of a model on new data

"""
train and test split

- split dataset into training and test datasets
- training dataset is used to train a model
- test dataset is used to measure model performance 

When comparing algorithms, use the same train/test datasets and seed 
the random number generator the same way before splitting the data.
Gives quick estimate. Only a single model created and evaluated. 
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
- the value of k should be divisible by the number of rows in the 
  training dataset so each k group has the same number of rows
- select a k value that allows for enough rows in each group so
  that each fold is representative of the original dataset
- rule of thumb: k=3 for small dataset, k=10 for large dataset
- calculate mean and std deviation on group and original - similar?
  the smaller they differ, the more representative the group

1) algorithm is trained 
2) algorithm is evaluated k times 
3) algorithm performance summarized by mean performance score
4) this is repeated k times so each k group can be the test dataset

If can, use. Cons: time-consuming to run if you have a large dataset
or are evaluating a model that take a long time to train. 
"""

"""
fold_size = count(rows) / count(folds) 
"""
def cross_validation_split(dataset, folds=3): 
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        # do not include kth fold in dataset split (training)
        while len(fold) < fold_size: 
            # random rows are removed from dataset copy 
            # and added to fold until fold size reached
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# example cross validation split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
folds = cross_validation_split(dataset, 4)
print(folds)
"""
[[[3], [2]], [[7], [1]], [[8], [9]], [[10], [6]]]
"""