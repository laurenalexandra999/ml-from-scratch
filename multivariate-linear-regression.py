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