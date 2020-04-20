"""
logistic regression

- input values X are combined to predict output Y
- the output value is binary (0 or 1)

yhat = 1.0 / (1.0 + e^-(b0 + b1 * X1))

- yhat is the predicted output
- b0 is the bias (intercept)
- b1 is the coefficient for the input X1 

- yhat is a value between 0 and 1 that is rounded to an integer
- that integer is mapped to a class value 

- each input value is associated with a b coefficient (a constant real value)
- b (beta) coefficients are learned from training data
- the representation of the model you would save is the coefficients
"""

"""
stochastic gradient descent

- gradient descent: minimize a function following the slope/gradient of the function
- SGD goal is minimize the error of a model by updating the coefficients each iteration

- b = b + learning_rate * (y - yhat) * yhat * (1 - yhat) * x

- b is the coefficient/weight being optimized
- learning rate you need to configure, e.g. 0.01
- (y - yhat) is the prediction error attributed to the coefficient
- yhat is the prediction made by the coefficients
- x is the input 
"""