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

"""
estimate coefficients

- learning rate: used to limit the amount each coefficient is corrected every time it is updated

- epochs: the number of times to run through the training data while updating the coefficients
"""

# estimate logistic regression coefficients with stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    # iterate over each epoch 
    for epoch in range(n_epoch):
        sum_error = 0
        # iterate over each row in training data for an epoch 
        for row in train:
            # predict
            yhat = predict(row, coef)
            # difference between expected and predicted value 
            error = row[-1] - yhat
            # sum of the squared error, a positive value 
            sum_error += error**2
            # the intercept coefficient is not responsible for an input value
            # b0(t + 1) = b0(t) + learning rate × (y(t) − yhat(t)) × yhat(t) × (1 − yhat(t))
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            # iterate over each coefficient and update it for a row in an epoch
            # there is one coefficient to weight each input value (row[i])
            for i in range(len(row) - 1):
                # b1(t + 1) = b1(t) + learning rate × (y(t) − yhat(t)) × yhat(t) × (1 − yhat(t)) × x1(t)
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef 
 
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

# learning rate 
l_rate = 0.3
# train the model for 100 epochs or exposures of the coefficients
n_epoch = 100

# calculate coefficients
coef = coefficients_sgd(dataset, l_rate, n_epoch)
"""
...
>epoch=97, lrate=0.300, error=0.023
>epoch=98, lrate=0.300, error=0.023
>epoch=99, lrate=0.300, error=0.022
"""
# the final set of coefficients 
print(coef)
# [-0.8596443546618897, 1.5223825112460005, -2.218700210565016]