"""
Biclustering algorithms.

"""

from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering

class Biclustering:

    def __init__(self):
        model = None


    @classmethod
    def spectral_biclustering(self, *args):
        """
            Wrapper method for the spectral_biclustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: 
        """
        self.model = SpectralBiclustering(*args)

    @classmethod
    def spectral_coclustering(self, *args):
        """
        Wrapper method for the spectral_coclustering algorithm
        :param args: the arguments to be sent to the sci-kit implementation
        :return: 
        """
        self.model = SpectralCoclustering(*args)

    def fit(self, data):
        """
        Performs biclustering
        :param data: Data to be fit
        :return: 
        """
        self.model.fit(data)

    def retrieve_bicluster(self, df, row_no, column_no):
        """
        Extracts the bicluster at the given row bicluster number and the column bicluster number from the input dataframe.
        :param df: the input dataframe whose values were biclustered
        :param row_no: the number of the row bicluster
        :param column_no: the number of the column bicluster
        :return: the extracted bicluster from the dataframe
        """
        res = df[self.model.biclusters_[0][row_no]]
        bicluster = res[res.columns[self.model.biclusters_[1][column_no]]]
        return bicluster

