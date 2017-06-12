"""
Clustering algorithms

"""

from sklearn.cluster import *

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

    @classmethod
    def hierarchical(cls, *args):
        """
            Wrapper method for the agglomerative clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = AgglomerativeClustering(*args)
        return cls(model)

    @classmethod
    def birch(cls, *args):
        """
            Wrapper method for the birch clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = Birch(*args)
        return cls(model)

    @classmethod
    def dbscan(cls, *args):
        """
            Wrapper method for the DBSCAN clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = DBSCAN(*args)
        return cls(model)

    @classmethod
    def feature_agglomeration(cls, *args):
        """
            Wrapper method for the feature agglomeration clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = FeatureAgglomeration(*args)
        return cls(model)

    @classmethod
    def mini_batch_kmeans(cls, *args):
        """
            Wrapper method for the mini batch k-means clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = MiniBatchKMeans(*args)
        return cls(model)

    @classmethod
    def mean_shift(cls, *args):
        """
            Wrapper method for the mean shift clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = MeanShift(*args)
        return cls(model)

    @classmethod
    def spectral_clustering(cls, *args):
        """
            Wrapper method for the spectral clustering algorithm
            :param args: the arguments to be sent to the sci-kit implementation
            :return: returns the Clustering object
        """

        model = SpectralClustering(*args)
        return cls(model)

    def fit(self, data):
        """
        Performs clustering
        :param data: Data to be fit
        :return: 
        """
        self.model.fit(data)

