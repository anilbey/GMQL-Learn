"""
Clustering algorithms

"""
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation


class Clustering:

    def __init__(self, model):
        self.model = model

    @classmethod
    def kmeans(cls, *args):
        """
            Wrapper method for the k-means clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = KMeans(*args)
        return cls(model)

    @classmethod
    def affinity_propagation(cls, *args):
        """
            Wrapper method for the affinity propagation clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = AffinityPropagation(*args)
        return cls(model)

    def fit(self, data):
        """
        Performs clustering
        :param data: Data to be fit
        :return: 
        """
        self.model.fit(data)

