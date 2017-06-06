import pandas as pd
from parser.parser import Parser
import numpy as np
import warnings


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

    def load(self, _path, regs=['chr', 'left', 'right', 'strand'], meta=[], values=[], full_load=False):
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
        self.data = p.parse_data(regs, values, full_load=full_load)

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
                self.data[v] = self.data[v].map(float)
        else:
            self.data[values] = self.data[values].map(float)
        print("started pivoting")
        self.data = pd.pivot_table(self.data,
                                   values=values, columns=selected_regions, index=['sample'],
                                   fill_value=default_value)
        print("end of pivoting")

