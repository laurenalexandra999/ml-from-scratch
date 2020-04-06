# reader() in csv module takes a file as an argument.
from csv import reader

# load a CSV file
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    # dataset is a list of lists. first list is rows (observations). second list is column values for a row.
    return dataset 

# convert string column values to floating point values
def str_column_to_float(dataset, column):
    for row in dataset:
        # strip value of whitespace before float conversion
        row[column] = float(row[column].strip())

# convert string column values to integer values
# some algorithms prefer the outcome value as numeric rather than string
def str_column_to_int(dataset, column):
    # obtain class values
    class_values = [row[column] for row in dataset]
    # py set - unordered iterable, mutable collection w/o duplicates
    unique = set(class_values)
    # py dictionary - unordered, mutable, indexed collection of key/value pairs
    # create dictionary of integer to class value map
    lookup = dict()
    # py enumerate(iterable, start=0) adds counter to an iterable and returns an enumerate object
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset: 
        # replace class value with integer value in dataset
        row[column] = lookup[row[column]]
    return lookup 

# load dataset 
filename = 'data.csv'
dataset = load_csv(filename) 
print('Loaded {0} with {1} rows and {2} columns.'.format(filename, len(dataset), len(dataset[0])))
# Loaded data.csv with 100 rows and 5 columns.

# convert string columns to float
# for each value in column range
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
print(dataset[0])

# convert class column values to integer values
lookup = str_column_to_int(dataset, 4) 
print(dataset[0])
print(lookup)