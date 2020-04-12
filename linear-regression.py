"""
linear regression 

- linear relationship between input and output. X --> Y
- simple linear regression (single input variable)
- Y = b0 + b1 * X
    - b0 and b1 are coefficients we need to estimate from training data
    - use these estimates to predict Y given new input X 
"""

"""
step 1: estimate mean and variance of input and output variables.
"""

def mean(values): 
    return sum(values) / float(len(values))

# variance is the sum squared difference for each value from the mean value
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

# example 
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x, mean_x), variance(y, mean_y)
print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y)) 
# x stats: mean=3.000 variance=10.000
# y stats: mean=2.800 variance=8.800 

"""
step 2: calculate covariance. 

- covariance describes how two or more groups of numbers change together
"""

# find covariance between X and Y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar 

# example 
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y) 
covar = covariance(x, mean_x, y, mean_y)
print('Covariance: %.3f' % (covar))
# Covariance: 8.000

"""
step 3: estimate coefficients. 

- b1 = covariance(x, y) / variance(x)
- b0 = mean(y) - (b1 * mean(x))
"""

def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean 
    return [b0, b1]

# example 
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
b0, b1 = coefficients(dataset)
print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))
# Coefficients: B0=0.400, B1=0.800 