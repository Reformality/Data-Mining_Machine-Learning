import os

import numpy as np
import pandas as pd


def shuffle_data(X, y, random_state=None):
    """
    Shuffle the data.

    Args:
        X: numpy array of shape (n, d)
        y: numpy array of shape (n, )
        seed: int or None

    Returns:
        X: shuffled data
        y: shuffled labels
    """
    if random_state:
        np.random.seed(random_state)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


def my_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):
    """
    Split the data into training and test sets.

    Args:
        X: numpy array of shape (N, D)
        y: numpy array of shape (N, )
        test_size: float, percentage of data to use as test set
        shuffle: bool, whether to shuffle the data or not
        seed: int or None

    Returns:
        X_train: numpy array of shape (N_train, D)
        X_val: numpy array of shape (N_val, D)
        y_train: numpy array of shape (N_train, )
        y_val: numpy array of shape (N_val, )
    """

    if shuffle:
        X, y = shuffle_data(X, y, random_state)

    n_train_samples = int(X.shape[0] * (1-test_size))
    # >> YOUR CODE HERE
    X_train, X_test = ...
    y_train, y_test = ...
    # END OF YOUR CODE <<

    return X_train, X_test, y_train, y_test


def load_phone_data(path: str, random_state=42):
    """
    Load the dataset. After loading, shuffle the dataset and replace the gender with binary coding.

    Args:
        path: path to the dataset

    Returns:
        X: data
        y: labels
    """

    df = pd.read_csv(path).sample(
        frac=1, random_state=random_state).reset_index(drop=True)

    X = df.drop(columns=['price_range']).values
    y = df['price_range'].values

    print(
        f'Loaded data from {path}:\n\r X dimension: {X.shape}, y dimension: {y.shape}')

    return X, y


def load_yelp_data(path: str):
    """
    Load the csv file and return both features and labels.
    The column "priceRange" is the label column, the possible values are 1, 2, 3, 4.
    All other columns are features columns. Their categorical values are listed in FEATURE_TABLE.

    Return:
            features (numpy 2d array), labels (numpy 1d array)
    """
    data = pd.read_csv(os.path.join(os.path.dirname(
        __file__), path)).fillna("NA").astype(str)

    # The first column is the label (priceRange)
    y = data.iloc[:, 0].values

    # The remaining columns are the features
    X = data.iloc[:, 1:].values

    return X, y


def accuracy(y, y_pred):
    """
    Calculate the accuracy of the predictions.

    Args:
        y: true labels
        y_pred: predicted labels

    Returns:
        accuracy: accuracy of the predictions
    """

    # >> YOUR CODE HERE
    return np.mean(y == y_pred)
    # << END OF YOUR CODE
