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