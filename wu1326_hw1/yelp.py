import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Yelp:
    def __init__(self, filename, sample_size=None, seed: int = 42) -> None:
        """
        Class constructor. DO NOT MODIFY.

        Args:
            filename: The name of the file containing the data set
            sample_size: The size of the data set to be used, None as default
            seed: The seed for the random number generator

        Returns:
            None
        """

        if sample_size is not None:
            np.random.seed(seed)
            self.df = self.read_data(filename).sample(sample_size)
        else:
            self.df = self.read_data(filename)

    def read_data(self, filename: str) -> pd.DataFrame:
        """
        Reads the data from the given file path.

        Args:
            filename: The path to the file to be read

        Returns:
            A pandas DataFrame containing the data

        """

        # >> YOUR CODE HERE
        return pd.read_csv(filename, header=0)
        # END OF YOUR CODE <<

    def average_rating(self) -> np.array:
        """
        Calculates the average star ratings of restaurants in each state, and then save the ratings in a lists in alphabetical 
        order of the state abbreviation, starting with AZ in ascending order.

        Returns:
            A np.array containing the average star ratings of restaurants.
        """

        # >> YOUR CODE HERE
        a = self.df.groupby(['state']).mean()
        return np.asarray(a[['stars']])
        # END OF YOUR CODE <<

    def rating_stats_given_review_count(self, count: int) -> tuple:
        """
        Calculates the mean and the standard deviation of the star ratings of restaurants that have at least that many ratings.

        Args:
            count: The minimum number of reviews for a restaurant to be included in the calculation

        Returns:
            A tuple of two floats representing the mean and standard deviation.
        """

        # >> YOUR CODE HERE

        a = self.df
        b = a[a['reviewCount'] >= count]
        mean = b['stars'].mean()
        std = b['stars'].std()

        # END OF YOUR CODE <<

        return float(mean), float(std)

    def plot_cdf(self, state: str = "NV") -> plt.Figure:
        """Generate plot of CDF of review counts for restaurants"""
        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)

        # >> YOUR CODE HERE
        a = self.df
        b = a[a['state'] == state]
        d = b.groupby(['reviewCount'])['state'].count().reset_index()
        y = np.array(d['state'])
        y = np.cumsum(y)
        y = np.divide(y,np.amax(y))
        x = np.array(d['reviewCount'])
        ax.plot(x, y)
        ax.set_xscale('log')
        ax.set_xlabel('log(reviewCount)')
        ax.set_ylabel('population')
        ax.set_title('CDF of review counts for NV')
        # END OF YOUR CODE <<
        return fig

    def make_boxplots(self) -> plt.Figure:
        """Create boxplots with distribution of number of checkins for each star rating level"""
        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)

        # >> YOUR CODE HERE

        ax.boxplot(..., ...)

        # END OF YOUR CODE <<

        return fig


"""
-------------------------------------------------------------------------------------------
THE CODE BELOW IS FOR EVALUATION. PLEASE DO NOT CHANGE!
-------------------------------------------------------------------------------------------
"""

from random import seed
import os

    
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

    # fig = yelp.make_boxplots()
    # fig.savefig(os.path.join(os.path.dirname(__file__), "yelp_boxplots.png"))

    print('Test yelp.py: passed')


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')
    evaluate_yelp()
