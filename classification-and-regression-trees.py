"""
CART - Classification and Regression Trees

- decision tree algorithms
- the CART model is a binary tree
- a node has an input X and a split point on a numeric X
- the leaf/terminal nodes have the output Y 
- this model can be navigated with new rows of data along each branch
  until a prediction is reached

- creating a binary tree requires splitting up the inputs with 
  a greedy approach, e.g. recursive binary splitting 
- this approach lines up all inputs and different split points
  are tested with a cost function
- the split point with the best cost (minimum cost) is chosen

- Regression: the cost function is the sum squared error
- Classification: the cost function is Gini, which tells how 
  pure the nodes are, i.e. how mixed up the training data assigned
  to each nodes are 
"""

"""
Gini Index

- a cost function to divide data into two groups of rows (a split)
- perfect score is 0. 
- worst case is a split with 50/50 classes in each group i.e. poor classification. score of 0.5.

gini_index = (1 - sum(proportion)) * (group_size / total_samples)

proportion = count(class_value) / count(rows)

e.g. a perfect split has 2 groups of data with 2 rows in each group.
the rows in the first group are class 0
the rows in the second group are class 1

group_1_class_0 = 2/2 = 1
group_1_class_1 = 0/2 = 0
group_2_class_0 = 0/2 = 0
group_2_class_1 = 2/2 = 1

Gini(group_1) = (1 - (1 * 1 + 0 * 0)) * (2/4)
Gini(group_1) = 0.0 * 0.5
Gini(group_1) = 0.0 
Gini(group_2) = (1 - (0 * 0 + 1 * 1)) * (2/4)
Gini(group_2) = 0.0 * 0.5
Gini(group_2) = 0.0

Perfect score: 
group_1 + group_2 = 0.0. 
"""

# calculate gini index
def gini_index(groups, classes):
    # count samples at split
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted index for each group
    gini = 0.0 
    for group in groups: 
        size = float(len(group))
        # do not divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group with each class score 
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its size
        gini += (1.0 - score) * (size / n_instances)
    return gini 

# example
print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
# 0.5
# 0.0

# splitting a dataset into two lists of rows given the index of an attribute
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right