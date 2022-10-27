import os
from os import path
from random import seed

import numpy as np

from decision_tree import *
from gain import *
from knn import KNN
from numpy_basics import *
from utils import *
from yelp import *

"""
evaluation.py is for evaluating the performance of your KNN and decision tree
model on the training and validation sets available to you. You should not
change this file. You can use this program to test the accuracy of your model
by calling it in this way:
    from evaluation import evaluate
    evaluate(o_train, p_train, o_valid, p_valid)
In this sample code, o_train is the original labels for the training set,
p_train is the predictions for the training set, o_valid is the original labels
for the validation set, and p_valid is the predictions for the validation set.
You can also directly run this program, without importing it, by calling it
with the following command:
    $ python evaluation.py
This will print out the accuracy of your model on the training set and the
validation set. It will also generate a plot of the loss and accuracy curves
for the training and validation sets.
"""

def accuracy(original, predictions):
    """
    Calculate the accuracy of given predictions on the given labels.

    Args:
        original: The original labels of shape (N,).
        predictions: Predictions of shape (N,).
        
    Returns:
        accuracy: The accuracy of the predictions.
    """
    # >> YOUR CODE HERE
    return ...
    # END OF YOUR CODE <<

def test(func, expected_output, **kwargs) -> bool:
    """
    Test a function with some inputs.

    Args:
        func: The function to test.
        expected_output: The expected output of the function.
        **kwargs: The arguments to pass to the function.

    Returns:
        True if the function outputs the expected output, False otherwise.
    """
    output = func(**kwargs)

    try:
        assert np.allclose(output, expected_output)
        print(f'Testing {func.__name__}: passed')
        return True
    except AssertionError:
        print(f'Testing {func.__name__}: failed')
        print(f'Expected:\n {expected_output}')
        print(f'Got:\n {output}')
        return False

def evaluate(o_train, p_train, o_valid=None, p_valid=None, o_test=None, p_test=None):
    """
    Calculate the accuracy of given predictions on the given labels.

    Args:
        o_train: The original labels of the training set of shape (N,).
        p_train: Predictions for the training set of shape (N,).
        o_valid: The original labels of the validation set of shape (N,).
        p_valid: Predictions for the validation set of shape (N,).
        o_test: The original labels of the test set of shape (N,), optional.
        p_test: Predictions for the test set of shape (N,), optional.

    Returns:
        None
    """

    print('\tTraining Accuracy:', accuracy(o_train, p_train))

    if o_valid is not None and p_valid is not None:
        print('\tValidation Accuracy:', accuracy(o_valid, p_valid))

    if o_test is not None and p_test is not None:
        print('\tTest Accuracy:', accuracy(o_test, p_test))
    else:
        print('\tTest Accuracy: Not available')

def evaluate_numpy_basics():
    """
    Test your implementation in numpy_basics.

    Args:
        None

    Returns:
        None
    """

    print('\n\n-------------Numpy Basics-------------\n')
    print('This test is not exhaustive by any means. You should test your ')
    print('implementation by yourself.\n')

    test(NumpyBasics.create_zero_matrix, np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), rows=3, cols=4)

    seed(42)
    test(NumpyBasics.create_vector, np.array([20, 3, 0, 23, 8]), n=5)

    test(NumpyBasics.calculate_matrix_inverse, np.array(
        [[-2, 1], [1.5, -0.5]]), matrix=np.array([[1, 2], [3, 4]]))

    test(NumpyBasics.calculate_dot_product, 32,
         vector1=np.array([1, 2, 3]), vector2=np.array([4, 5, 6]))

    test(NumpyBasics.solve_linear_system, np.array(
        [-4, 4.5]), A=np.array([[1, 2], [3, 4]]), b=np.array([5, 6]))


def evaluate_yelp():
    """
    Test your implementation in yelp.py.

    Args:
        None

    Returns:
        None
    """

    print('\n\n-------------Yelp Dataset-------------\n')
    print('This test is not exhaustive by any means. It only tests if')
    print('your implementation runs without errors.\n')

    yelp = Yelp(os.path.join(os.path.dirname(__file__), "dataset/yelp.csv"))

    fig = yelp.plot_cdf()
    fig.savefig(os.path.join(os.path.dirname(__file__), "yelp_cdf.png"))

    fig = yelp.make_boxplots()
    fig.savefig(os.path.join(os.path.dirname(__file__), "yelp_boxplots.png"))

    print('Test yelp.py: passed')

def evaluate_KNN() -> None:
    """
    Evaluate the KNN model on the data set.

    Args:
        None
        
    Returns:
        None
    """
    print('\n\n-------------KNN Performace-------------\n')
    X, y = load_phone_data(os.path.join(
        os.path.dirname(__file__), r'dataset/phone_train.csv'))
    X_train, X_valid, y_train, y_valid = my_train_test_split(X, y)

    # Test set that will be used to evaluate your model output on unseen data is not provided.
    X_test, y_test = load_phone_data(os.path.join(
        os.path.dirname(__file__), r'dataset/phone_test.csv'))

    # Initialize and fit a KNN model.
    knn = KNN(k=5)
    knn.fit(X_train, y_train)

    # Evaluate the KNN model on the training set.

    evaluate(y_train, knn.predict(X_train), y_valid,
             knn.predict(X_valid), y_test, knn.predict(X_test))

def evaluate_DecisionTrees():
    """
    Evaluate the Decision Trees performance on the data set.

    Args:
        None
        
    Returns:
        None
    """
    X, y = load_yelp_data(
		path.join(path.dirname(__file__), "dataset/yelp_train.csv")
	)
    scorer = GiniGain(class_labels=set(y))
    model, loss, _ = decision_tree_loss(X, y, X, y, scorer, max_depth=1)
    print("0-1 Loss:", loss)
    print(model)
    # Test set that will be used to evaluate your model output on unseen data is not provided.
    print('On Test Dataset')
    X, y = load_yelp_data(
		path.join(path.dirname(__file__), "dataset/yelp_test.csv")
	)
    scorer = GiniGain(class_labels=set(y))
    model, loss, _ = decision_tree_loss(X, y, X, y, scorer, max_depth=1)
    print("0-1 Loss:", loss)
    print(model)



if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    evaluate_numpy_basics()

    evaluate_yelp()
    
    evaluate_KNN()

    evaluate_DecisionTrees() 

    print('\n\nDone.')
