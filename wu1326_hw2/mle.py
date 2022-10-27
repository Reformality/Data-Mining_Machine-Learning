import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi


class Gaussian:
    def get_data(self, mean: int, std: int, size: int, seed: int = 42) -> np.array:
        """
        Create a dataset with each value drawn from a Gaussian distribution

        Args:
            mean: The mean of the data set
            std: The standard deviation of the data set
            size: The size of the data set
            seed: A parameter that controls the randomness. DON'T MODIFY!

        Returns:
            data: A numpy array of size size, with values being independently
                and identically Gaussian distributed
        """

        np.random.seed(seed)
        data = np.zeros(size)

        # >> YOUR CODE HERE
        data = np.random.normal(mean, std, size)
        # << END OF YOUR CODE

        return data

    def log_pdf(self, x: np.array, mean: float, var: float) -> np.array:
        """
        Calculates the log of the PDF of the data set

        Args:
            x: The data set for which the probability density function is to 
                be calculated
            mean: The mean of the data sets
            var: The square of the standard deviation of the data set

        Returns:
            A numpy array of size x.size, with each element being the log of
            the PDF of the data set at the corresponding element of x
        """

        # >> YOUR CODE HERE
        log_pdf = (1/np.sqrt(2*pi*var*var))*np.exp(-(x-mean)**2/(2*var*var))
        
        # << END OF YOUR CODE
        return log_pdf

    def histogram(self, x: np.array) -> plt.Figure:
        """
        Plots a histogram of the data set. Add a vertical line to the plot at
        the mean of the data set.

        Args:
            x: The data set for which the histogram is to be plotted

        Returns:
            A matplotlib figure object
        """

        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
        
        # >> YOUR CODE HERE
        ax.hist(x)

        # this is for plotting the vertical line of estimated mean value. Please learn how to 
        # use it by yourself

        ax.axvline(x=np.mean(x), color='r')
        # << END OF YOUR CODE

        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of Gaussian Distribution")

        return fig

    def mle(self, x: np.array) -> np.array:
        """
        Calculates the maximum likelihood estimate of the parameters of a
        Gaussian distribution.

        Args:
            x: The data set for which the MLE is to be calculated

        Returns:
            A numpy array of size 2, with the first element being the mean and
            the second element being the square of the standard deviation. You 
            may use the result from the theoretical part.
        """

        # >> YOUR CODE HERE
        mean = np.mean(x)
        var = np.var(x)
        # << END OF YOUR CODE
        return np.array([mean, var])

    def gradient(self, x, theta) -> np.array:
        """
        Calculates the gradient of the log likelihood function with respect to
        the parameters of the Gaussian distribution. The gradient will be
        used in gradient descent to find the optimal parameters.

        Args:
            x: The data set for which the gradient is to be calculated
            theta: The parameters to be used in the log likelihood calculation,
                    theta[0] is the mean, theta[1] is the square of the
                    standard deviation

        Returns:
            A numpy array of size 2, with the first element being the gradient
            of the log likelihood with respect to the mean and the second element
            being the gradient of the log likelihood with respect to the square
            of the standard deviation. 
        """

        grad = np.zeros(2)
        mean = theta[0]
        var = theta[1]

        # if var < 0:
        #     var = 1e-5

        # >> YOUR CODE HERE

        # gradient of mean
        grad[0] = 0
        for val in x:
            grad[0] = grad[0] + (val-mean)
        grad[0] = grad[0] * (1/var)

        # gradient of var
        grad[1] = 0
        for val2 in x:
            grad[1] = grad[1] + (val2-mean)**2
        grad[1] = grad[1] * (1/(var*np.sqrt(var))) + (-x.size / np.sqrt(var))

        # << END OF YOUR CODE
        return -grad

    def fit(self,
            x: np.array,
            init_theta: np.array,
            max_epochs: int = 10**5,
            learning_rate: float = 1e-4,
            tolerance: float = 1e-3,
            print_progress=True) -> np.array:
        """
        Finds the maximum likelihood estimate of the parameters of a Gaussian
        distribution using the gradient descent algorithm.

        Args:
            x: The data set for which the MLE is to be calculated
            init_theta: The initial value for the parameters of the Gaussian
                    distribution
            max_epochs: The maximum number of epochs to run the algorithm
            learning_rate: The learning rate to be used in the gradient descent
            tolerance: The tolerance to be used in the gradient descent algorithm
                        to determine when the algorithm has converged

        Returns:
            A numpy array of size 2, with the first element being the mean and
            the second element being the square of the standard deviation.
        """

        theta = init_theta

        for i in range(max_epochs):
            # >> YOUR CODE HERE
            grad = self.gradient(x, theta)
            theta = theta - learning_rate * grad
            # print(grad, theta)
            # print(np.linalg.norm(grad))
            # << END OF YOUR CODE

            if i % 100 == 0 and print_progress:
                print(
                    f'Epoch {i}: Mean = {theta[0]:.8f}, Variance = {theta[1]:.8f}')
            if np.linalg.norm(grad) < tolerance:
                print(f'Converged after {i} epochs')
                break

        return theta

"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""

def evaluate_Gaussian_MLE(print_progress=True):
    """
    Test your implementation of Gaussian MLE.

    Args:
        print_progress: Whether to print the progress of the fitting process.

    Returns:
        None
    """
    print('This test is not exhaustive by any means. It only tests if')
    print('your implementation runs without errors.\n')
    gaussian_mle = Gaussian()

    mean, std = 0, 1

    data = gaussian_mle.get_data(mean=mean, std=std, size=10000)

    gaussian_mle.histogram(data).savefig(__file__.split(".")[0] + "_hist.png")

    mle_params = gaussian_mle.mle(data)
    init_gd_params = np.array([4, 100])

    gd_params = gaussian_mle.fit(
        data, init_gd_params, print_progress=print_progress)

    print('\nParameters estimated by gradient descent:')
    print(
        f'\tMean: {gd_params[0]:.5f},\tVariance: {gd_params[1]:.5f}')

    print('Parameters estimated by MLE:')
    print(
        f'\tMean: {mle_params[0]:.5f},\tVariance: {mle_params[1]:.5f}')
    print('Parameters used to generate data:')
    print(f'\tMean: {mean:.5f}, \tVariance: {std**2:.5f}')


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    print('\n-------------Gaussian MLE-------------')
    evaluate_Gaussian_MLE(print_progress=False)
