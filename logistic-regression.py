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
from math import exp 

# predict an output for a row given coefficients
def predict(row, coefficients):
    # the bias/intercept is not responsible for an input value
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        # y = b0 + b1 * x 
        yhat += coefficients[i + 1] * row[i]
    # yhat = 1.0 / (1.0 + e^-(b0 + b1 * X1))
    return 1.0 / (1.0 + exp(-yhat)) 

dataset = [[2.7810836,2.550537003,0],
[1.465489372,2.362125076,0],
[3.396561688,4.400293529,0],
[1.38807019,1.850220317,0],
[3.06407232,3.005305973,0],
[7.627531214,2.759262235,1],
[5.332441248,2.088626775,1],
[6.922596716,1.77106367,1],
[8.675418651,-0.242068655,1],
[7.673756466,3.508563011,1]]
coef = [-0.406605464, 0.852573316, -1.104746259]

for row in dataset: 
    yhat = predict(row, coef)
    print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat))) 

"""
Expected=0.000, Predicted=0.299 [0]
Expected=0.000, Predicted=0.146 [0]
Expected=0.000, Predicted=0.085 [0]
Expected=0.000, Predicted=0.220 [0]
Expected=0.000, Predicted=0.247 [0]
Expected=1.000, Predicted=0.955 [1]
Expected=1.000, Predicted=0.862 [1]
Expected=1.000, Predicted=0.972 [1]
Expected=1.000, Predicted=0.999 [1]
Expected=1.000, Predicted=0.905 [1]
"""