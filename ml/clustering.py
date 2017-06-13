"""
Clustering algorithms

"""

from sklearn.cluster import *
from sklearn.metrics.cluster import *

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

    def retrieve_cluster(self, df, cluster_no):
        """
        Extracts the cluster at the given index from the input dataframe
        :param df: the dataframe that contains the clusters
        :param cluster_no: the cluster number
        :return: returns the extracted cluster
        """
        mask = self.model.labels_ == cluster_no  # a boolean mask
        return df[mask]

    def adjusted_mutual_info(self, reference_clusters):
        """
        Calculates the adjusted mutual information score w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: returns the value of the adjusted mutual information score
        """
        return adjusted_mutual_info_score(self.model.labels_, reference_clusters.model.labels_)

    def adjusted_rand_score(self, reference_clusters):
        """
        Calculates the adjusted rand score w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: returns the value of the adjusted rand score
        """
        return adjusted_rand_score(self.model.labels_, reference_clusters.model.labels_)

    def calinski_harabasz(self, data):
        """
        Calculates the Calinski-Harabarsz score for a set of clusters (implicit evaluation).
        :param data: The original dataset that the clusters are generated from
        :return: The resulting Calinski-Harabarsz score
        """
        return calinski_harabaz_score(data, self.model.labels_)

    def completeness_score(self, reference_clusters):
        """
        Calculates the completeness score w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: the resulting completeness score
        """
        return completeness_score(self.model.labels_, reference_clusters.model.labels_)

    def fowlkes_mallows(self, reference_clusters):
        """
        Calculates the Fowlkes-Mallows index (FMI) w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: The resulting Fowlkes-Mallows score.
        """
        return fowlkes_mallows_score(self.model.labels_, reference_clusters.model.labels_)

    def homogeneity_score(self, reference_clusters):
        """
        Calculates the homogeneity score w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: The resulting homogeneity score.
        """
        return homogeneity_score(self.model.labels_, reference_clusters.model.labels_)

    def mutual_info_score(self, reference_clusters):
        """
        Calculates the MI (mutual information) w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: The resulting MI score.
        """
        return mutual_info_score(self.model.labels_, reference_clusters.model.labels_)

    def normalized_mutual_info_score(self, reference_clusters):
        """
        Calculates the normalized mutual information w.r.t. the reference clusters (explicit evaluation)
        :param reference_clusters: Clusters that are to be used as reference  
        :return: The resulting normalized mutual information score.
        """
        return normalized_mutual_info_score(self.model.labels_, reference_clusters.model.labels_)

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
        return silhouette_score(data, self.model.labels_, metric, sample_size, random_state, **kwds)