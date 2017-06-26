import pandas as pd
from parser.parser import Parser
import numpy as np
import warnings
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
from ml.clustering import Clustering
from ml.biclustering import Biclustering
from tqdm import tqdm

class DataModel:
    """
    The Data Model for the manipulation of GDM data
    """

    def __init__(self):
        """
        Constructor
        """
        self.data = None
        self.meta = None
        self._path = None
        return

    @classmethod
    def from_memory(cls, data, meta):
        """
        Overloaded constructor to create the DataModel object from memory data and meta variables.
        Args:
            :param data: The data model
            :param meta: The metadata
            :return: A DataModel object
        """

        obj = cls()
        obj.data = data
        obj.meta = meta
        return obj

    def load(self, _path, regs=['chr', 'left', 'right', 'strand'], meta=[], values=[], full_load=False, file_extension="gdm"):
        """Parses and loads the data into instance attributes.

        Args:
            :param path: The path to the dataset on the filesystem
            :param regs: the regions that are to be analyzed
            :param meta: the meta-data that are to be analyzed
            :param values: the values that are to be selected
            :param full_load: Specifies the method of parsing the data. If False then parser omits the parsing of zero(0)
            values in order to speed up and save memory. However, while creating the matrix, those zero values are going to be put into the matrix.
            (unless a row contains "all zero columns". This parsing is strongly recommended for sparse datasets.
            If the full_load parameter is True then all the zero(0) data are going to be read.

        """
        if not full_load:
            warnings.warn("\n\nYou are using the optimized loading technique. "
                          "All-zero rows are not going to be loaded into memory. "
                          "To load all the data please set the full_load parameter equal to True.")
        p = Parser(_path)
        self.meta = p.parse_meta(meta)
        self.data = p.parse_data(regs, values, full_load=full_load, extension=file_extension)
        self._path = _path

    def set_meta(self, selected_meta):
        """Sets one axis of the 2D multi-indexed dataframe
            index to the selected meta data.

        Args:
            :param selected_meta: The list of the metadata users want to index with.

        """
        meta_names = list(selected_meta)
        meta_names.append('sample')
        meta_index = []
        # To set the index for existing samples in the region dataframe.
        # The index size of the region dataframe does not necessarily be equal to that of metadata df.
        warnings.warn("\n\nThis method assumes that the last level of the index is the sample_id.\n"
                      "In case of single index, the index itself should be the sample_id")
        for x in meta_names:
            meta_index.append(self.meta.ix[self.data.index.get_level_values(-1)][x].values)
        meta_index = np.asarray(meta_index)
        multi_meta_index = pd.MultiIndex.from_arrays(meta_index, names=meta_names)
        self.data.index = multi_meta_index

    def to_matrix(self, values, selected_regions, default_value=0):
        """Creates a 2D multi-indexed matrix representation of the data.
            This representation allows the data to be sent to the machine learning algorithms.

        Args:
            :param values: The value or values that are going to fill the matrix.
            :param selected_regions: The index to one axis of the matrix.
            :param default_value: The default fill value of the matrix

        """
        if isinstance(values, list):
            for v in values:
                try:
                    self.data[v] = self.data[v].map(float)
                except:
                    print(self.data[v])
        else:
            self.data[values] = self.data[values].map(float)
        print("started pivoting")
        self.data = pd.pivot_table(self.data,
                                   values=values, columns=selected_regions, index=['sample'],
                                   fill_value=default_value)
        print("end of pivoting")


    def get_values(self, cluster, selected_meta):
        """
        Retrieves the values which are later going to be used in the calculation of statistics
        :param cluster: cluster that contains the data
        :param selected_meta: the values of the selected_meta
        :return: the values of the selected meta of the cluster
        """
        warnings.warn("\n\nThis method assumes that the last level of the index is the sample_id.\n"
                      "In case of single index, the index itself should be the sample_id")
        sample_ids = cluster.index.get_level_values(-1)
        corresponding_meta = self.meta.loc[sample_ids]
        values = corresponding_meta[selected_meta]

        try:
            values = values.astype(float)
        except ValueError:
            print("the values should be numeric")
        return values

    def cluster_mean(self, cluster, selected_meta):
        """
        calculates the mean of the cluster based on the selected meta data
        :param cluster: cluster that contains the data
        :param selected_meta: the selected meta data name
        :return: mean value of the selected meta of the cluster
        """
        values = self.get_values(cluster, selected_meta)
        mean = statistics.mean(values)
        print(mean)
        return mean

    def cluster_variance(self, cluster, selected_meta):
        """
        calculates the variance of the cluster based on the selected meta data
        :param cluster: cluster that contains the data
        :param selected_meta: the selected meta data name
        :return: variance of the selected meta of the cluster
        """
        values = self.get_values(cluster, selected_meta)
        var = statistics.variance(values)
        print(var)
        return var

    def cluster_std(self, cluster, selected_meta):
        """
            calculates the standard deviation of the cluster based on the selected meta data
            :param cluster: cluster that contains the data
            :param selected_meta: the selected meta data name
            :return: standard deviation of the selected meta of the cluster
        """
        values = self.get_values(cluster, selected_meta)
        stdev = statistics.stdev(values)
        print(stdev)
        return stdev

    def to_bag_of_genomes(self, clustering_object):
        """
        Creates a bag of genomes representation for data mining purposes
        Each document (genome) in the representation is a set of metadata key and value pairs belonging to the same cluster.
        The bag of genomes are saved under ./bag_of_genomes/ directory
        :param clustering_object: The clustering object
        :return: None
        """

        meta_files = Parser._get_files('meta', self._path)
        meta_dict = {}
        for f in meta_files:
            meta_dict[Parser.get_sample_id(f)] = f

        clusters = []
        if isinstance(clustering_object, Clustering):
            no_clusters = clustering_object.model.n_clusters
            for c in range(0, no_clusters):
                clusters.append(clustering_object.retrieve_cluster(self.data, c).index.get_level_values(-1).values)

        elif isinstance(clustering_object, Biclustering):
            no_clusters = clustering_object.model.n_clusters[0]  # 0 for the rows
            no_col_clusters = clustering_object.model.n_clusters[1]  # 1 for the columns
            for c in range(0, no_clusters):
                clusters.append(clustering_object.retrieve_bicluster(self.data, c*no_col_clusters, 0).index.get_level_values(-1).values)  # sample names

        # to create the bag of genomes files
        print("creating the bag_of_genomes...")
        for c in tqdm(range(0, no_clusters)):
            document = open('./bag_of_genomes/document' + str(c) + '.bag_of_genome', 'w')
            for sample in clusters[c]:
                f = open(meta_dict[sample], 'r')
                for line in f:
                    line = line.replace(' ', '_')
                    splitted = line.split('\t')
                    document.write(splitted[0] + '=' + splitted[1])
                f.close()
            document.close()

    @staticmethod
    def to_term_document_matrix(path_to_bag_of_genomes, max_df=0.99, min_df=1):
        """
        Creates a term-document matrix which is a mathematical matrix that describes the frequency of terms
        that occur in a collection of documents (in our case a collection of genomes).
        :param path_to_bag_of_genomes: Path to the documents (genomes)
        :param max_df: To prune the terms that are existing in the given portion of documents (if set to 1 then it does not prune)
        :return: returns the term-document dataframe
        """
        token_dict = {}

        def BoG_tokenizer(_text):
            return _text.split('\n')

        print("creating the term-document matrix...")
        for file in tqdm(Parser._get_files('.bag_of_genome', path_to_bag_of_genomes)):
            f = open(file, 'r')
            text = f.read()
            token_dict[file] = text

        tfidf = TfidfVectorizer(tokenizer=BoG_tokenizer, use_idf=True, smooth_idf=False,
                                max_df=max_df, min_df=min_df)  # max df is less than 1.0 to ignore the tokens existing in all of the documents

        tfs = tfidf.fit_transform(token_dict.values())
        data = tfs.toarray()
        columns = tfidf.get_feature_names()
        df = pd.DataFrame(data, columns=columns)
        term_document_df = df.T
        return term_document_df
