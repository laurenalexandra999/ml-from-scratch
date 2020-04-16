"""
multivariate linear regression

- using input values to predict an output value: 
  each input attribute (x) is weighted using 
  a coefficient (b) with the goal of the ml algorithm 
  as finding coefficients that produce a good prediction (y)
  y = b0 + b1 * x1 + b2 * x2 + ...
"""

"""
stochastic gradient descent 

- optimization algorithms like stochastic gradient descent 
  are used by machine learning algorithms to identify a good
  set of model parameters with training data

- gradient descent: minimize a function following the slope or gradient
  of the function 

- evaluate and update the coefficients every iteration to minimize 
  the error of a model on training data 

- update the coefficients using: b = b - learning_rate * error * x
  - (b) is the coefficient/weight being optimized 
  - learning rate is a rate you need to configure (e.g. 0.01)
  - error is the model prediction error attributed to the coefficient
  - (x) is the input 
"""

"""
predict with coefficients

- the first coefficient is the intercept. this is the bias or b0. 
- y = b0 + b1 * x 
"""

def predict(row, coefficients):
    # hat - estimated value
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat 

# example
dataset  = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
coef = [0.4, 0.8]
for row in dataset:
    # y = 0.4 + 0.8 * x
    yhat = predict(row, coef)
    print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))
"""
Expected=1.000, Predicted=1.200
Expected=3.000, Predicted=2.000
Expected=3.000, Predicted=3.600
Expected=2.000, Predicted=2.800
Expected=5.000, Predicted=4.400
"""

"""
estimate linear regression coefficents using stochastic gradient descent

- stochastic gradient descent requires these parameters:
  - learning rate: limits the amount a coefficient is corrected
    each time the coefficient is updated
  - epochs: the count of times running through the training data
    while updating the coefficients
"""
def coefficents_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    # iterate over each epoch
    for epoch in range(n_epoch):
        sum_error = 0
        # iterate over each row in training data for a given epoch
        for row in train:
            yhat = predict(row, coef)
            # model error = prediction - expected
            error = yhat - row[-1]
            # sum of the squared error (positive value)
            sum_error += error**2
            # bias coefficient: b0(t + 1) = b0(t) - learning rate * error(t)
            coef[0] = coef[0] - l_rate * error
            # iterate over each coefficient and update it for each row in a given epoch
            for i in range(len(row) - 1):
                # b1(t + 1) = b1(t) - learning rate * error(t) * x1(t)
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef 

# example 
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
l_rate = 0.001
n_epoch = 50
coef = coefficents_sgd(dataset, l_rate, n_epoch)
"""
...
>epoch=45, lrate=0.001, error=2.650
>epoch=46, lrate=0.001, error=2.627
>epoch=47, lrate=0.001, error=2.607
>epoch=48, lrate=0.001, error=2.589
>epoch=49, lrate=0.001, error=2.573
[0.22998234937311363, 0.8017220304137576]
"""

"""
example linear regression on csv
"""
seed(1)
filename = 'data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
  str_column_to_float(dataset, i)
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
n_folds = 5
l_rate = 0.01
n_epoch = 50
scores = evalute_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))