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

# evaluating all splits: check every value on each attribute as
# a candidate split and evaluate the split cost 
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    # iterate over each attribute
    for index in range(len(dataset[0])-1):
        # iterate over each value for an attribute 
        for row in dataset:
            # split dataset into two lists of rows 
            groups = test_split(index, row[index], dataset)
            # use cost function to divide data into two groups of rows (a split)
            gini = gini_index(groups, class_values)
            print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
        if gini < b_score:
            b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    # the best split represented as a node in decision tree
    # index of attribute
    # value of attribute by which to split
    # two groups of data split by the splitting point 
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Example: Find the best split
dataset = [[2.771244718,1.784783929,0],
[1.728571309,1.169761413,0],
[3.678319846,2.81281357,0],
[3.961043357,2.61995032,0],
[2.999208922,2.209014212,0],
[7.497545867,3.162953546,1],
[9.00220326,3.339047188,1],
[7.444542326,0.476683375,1],
[10.12493903,3.234550982,1],
[6.642287351,3.319983761,1]]

split = get_split(dataset)

"""
X1 < 2.771 Gini=0.444
X1 < 1.729 Gini=0.500
X1 < 3.678 Gini=0.286
X1 < 3.961 Gini=0.167
X1 < 2.999 Gini=0.375
X1 < 7.498 Gini=0.286
X1 < 9.002 Gini=0.375
X1 < 7.445 Gini=0.167
X1 < 10.125 Gini=0.444
X1 < 6.642 Gini=0.000   # perfect split 
X2 < 1.785 Gini=0.500
X2 < 1.170 Gini=0.444
X2 < 2.813 Gini=0.320
X2 < 2.620 Gini=0.417
X2 < 2.209 Gini=0.476
X2 < 3.163 Gini=0.167
X2 < 3.339 Gini=0.444
X2 < 0.477 Gini=0.500
X2 < 3.235 Gini=0.286
X2 < 3.320 Gini=0.375
"""

print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
# Split: [X1 < 6.642]