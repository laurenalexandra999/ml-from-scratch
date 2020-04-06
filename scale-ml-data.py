# algorithms often expect the scale of the i/o data to be the same

"""
Normalization

- rescaling an input to range between 0 and 1
- need to know min and max values for each attribute in dataset
- estimate min and max values by enumerating through them

If the data is not normally distributed, consider normalizing the data.
"""

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax 

# rescale column values to range 0 and 1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            # scaled value = (value - min) / (max - min)
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

dataset = [[50, 30], [20, 90]] 
"""
col1    col2
50      30
20      90
"""
# estimate min and max values for each column
minmax = dataset_minmax(dataset) 
print(minmax) 
# [[20, 50], [30, 90]]

# given min and max estimates, normalize raw data to range 0 and 1
normalize_dataset(dataset, minmax)
print(dataset)
# [[1, 0], [0, 1]]

"""
Standardization 

- rescaling that centers the distribution of the data on the value 0 and the standard deviation to the value 1
- the mean and standard deviation can summarize a normal distribution aka Gaussian distribution, bell curve
- need to know mean and standard deviation before scaling 
- we can estimate mean and standard deviation for each column in training data

Is a scaling technique that assumes the distribution of the data is normal.
Use if a column attribute is normal/close to normal. 
"""
from math import sqrt 

def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means 

def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        # (value - mean)^2 
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset) - 1))) for x in stdevs]
    return stdevs  

def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            # standardized_value = (value - mean) / stdev 
            row[i] = (row[i] - means[i]) / stdevs[i]

means = column_means(dataset)
stdevs = column_stdevs(dataset, means)

# standardize column values
standardize_dataset(dataset, means, stdevs)
print(dataset) 
