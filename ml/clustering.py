"""
Clustering algorithms

"""

from sklearn.cluster import *
from sklearn.metrics.cluster import *
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.clarans import clarans
import pandas as pd

class Clustering:

    def __init__(self, model):
        self.model = model

    @classmethod
    def xmeans(cls, initial_centers=None, kmax=20, tolerance=0.025, criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore=False):
        """
        Wrapper method for the x-means clustering algorithm
        :param initial_centers: Initial coordinates of centers of clusters that are represented by list: [center1, center2, ...] 
        Note: The dimensions of the initial centers should be same as of the dataset. 
        :param kmax: Maximum number of clusters that can be allocated.
        :param tolerance: Stop condition for each iteration: if maximum value of change of centers of clusters is less than tolerance than algorithm will stop processing
        :param criterion: Type of splitting creation.
        :param ccore: Defines should be CCORE (C++ pyclustering library) used instead of Python code or not.
        :return: returns the clustering object
        """
        model = xmeans(None, initial_centers, kmax, tolerance, criterion, ccore)
        return cls(model)

    @classmethod
    def clarans(cls, number_clusters, num_local, max_neighbour):
        """
        Wrapper method for the CLARANS clustering algorithm
        :param number_clusters: the number of clusters to be allocated
        :param num_local: the number of local minima obtained (amount of iterations for solving the problem).
        :param max_neighbour: the number of local minima obtained (amount of iterations for solving the problem).
        :return: the resulting clustering object
        """
        model = clarans(None, number_clusters, num_local, max_neighbour)
        return cls(model)

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

    @staticmethod
    def is_pyclustering_instance(model):
        """
        Checks if the clustering algorithm belongs to pyclustering
        :param model: the clustering algorithm model
        :return: the truth value (Boolean)
        """
        return any(isinstance(model, i) for i in [xmeans, clarans])

    def fit(self, data):
        """
        Performs clustering
        :param data: Data to be fit
        :return: 
        """
        if self.is_pyclustering_instance(self.model):
            if isinstance(data, pd.DataFrame):
                data = data.values.tolist()
            else:  # in case data is already in the matrix form
                data = data.tolist()
            if isinstance(self.model, xmeans):
                self.model._xmeans__pointer_data = data
            elif isinstance(self.model, clarans):
                self.model._clarans__pointer_data = data
            self.model.process()
        else:
            self.model.fit(data)

    @property
    def _labels_from_pyclusters(self):
        """
        Computes and returns the list of labels indicating the data points and the corresponding cluster ids.
        :return: The list of labels
        """
        clusters = self.model.get_clusters()
        labels = []
        for i in range(0, len(clusters)):
            for j in clusters[i]:
                labels.insert(j, i)
        return labels

    def retrieve_cluster(self, df, cluster_no):
        """
        Extracts the cluster at the given index from the input dataframe
        :param df: the dataframe that contains the clusters
        :param cluster_no: the cluster number
        :return: returns the extracted cluster
        """
        if self.is_pyclustering_instance(self.model):
            clusters = self.model.get_clusters()
            mask = []
            for i in range(0, df.shape[0]):
                mask.append(i in clusters[cluster_no])
        else:
            mask = self.model.labels_ == cluster_no  # a boolean mask
        return df[mask]

    @staticmethod
    def get_labels(obj):
        """
        Retrieve the labels of a clustering object
        :param obj: the clustering object
        :return: the resulting labels
        """
        if Clustering.is_pyclustering_instance(obj.model):
            return obj._labels_from_pyclusters
        else:
            return obj.model.labels_

    def adjusted_mutual_info(self, reference_clusters):
        """
        Calculates the adjusted mutual information score w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: returns the value of the adjusted mutual information score
        """
        return adjusted_mutual_info_score(self.get_labels(self), self.get_labels(reference_clusters))

    def adjusted_rand_score(self, reference_clusters):
        """
        Calculates the adjusted rand score w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: returns the value of the adjusted rand score
        """
        return adjusted_rand_score(self.get_labels(self), self.get_labels(reference_clusters))

    def calinski_harabasz(self, data):
        """
        Calculates the Calinski-Harabarsz score for a set of clusters (implicit evaluation).
        :param data: The original dataset that the clusters are generated from
        :return: The resulting Calinski-Harabarsz score
        """
        return calinski_harabaz_score(data, self.get_labels(self))

    def completeness_score(self, reference_clusters):
        """
        Calculates the completeness score w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: the resulting completeness score
        """
        return completeness_score(self.get_labels(self), self.get_labels(reference_clusters))

    def fowlkes_mallows(self, reference_clusters):
        """
        Calculates the Fowlkes-Mallows index (FMI) w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: The resulting Fowlkes-Mallows score.
        """
        return fowlkes_mallows_score(self.get_labels(self), self.get_labels(reference_clusters))

    def homogeneity_score(self, reference_clusters):
        """
        Calculates the homogeneity score w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: The resulting homogeneity score.
        """
        return homogeneity_score(self.get_labels(self), self.get_labels(reference_clusters))

    def mutual_info_score(self, reference_clusters):
        """
        Calculates the MI (mutual information) w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: The resulting MI score.
        """
        return mutual_info_score(self.get_labels(self), self.get_labels(reference_clusters))

    def normalized_mutual_info_score(self, reference_clusters):
        """
        Calculates the normalized mutual information w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: The resulting normalized mutual information score.
        """
        return normalized_mutual_info_score(self.get_labels(self), self.get_labels(reference_clusters))

    def silhouette_score(self, data,  metric='euclidean', sample_size=None, random_state=None, **kwds):
        """
        Computes the mean Silhouette Coefficient of all samples (implicit evaluation)
        :param data: The data that the clusters are generated from
        :param metric: the pairwise distance metric
        :param sample_size: the size of the sample to use computing the Silhouette Coefficient
        :param random_state: If an integer is given then it fixes its seed otherwise random.
        :param kwds: any further parameters that are passed to the distance function
        :return: the mean Silhouette Coefficient of all samples
        """
        return silhouette_score(data, self.get_labels(self), metric, sample_size, random_state, **kwds)