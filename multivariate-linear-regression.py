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