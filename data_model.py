import pandas as pd
from parser.parser import Parser
import numpy as np


class DataModel:
    """
    The Data Model for the manipulation of GDM data
    """

    def __init__(self):
        self.data = None
        self.meta = None
        return

    @classmethod
    def from_memory(cls, data, meta):
        obj = cls()
        obj.data = data
        obj.meta = meta
        return obj

    def load(self, _path, regs=['chr', 'left', 'right', 'strand'], meta=[], values=[], full_load=False):
        """Parses and loads the data into instance attributes.

        Args:
            :param path: The path to the dataset on the filesystem.
            regs: the regions that are to be analyzed.
            meta: the meta-data that are to be analyzed.
            values: the values that are to be selected.
            full_load: if true then the all-zero rows are also read

        """
        p = Parser(_path)
        self.meta = p.parse_meta(meta)
        self.data = p.parse_data(regs, values, full_load=full_load)

    def set_meta(self, selected_meta):
        """Sets one axis of the 2D multi-indexed dataframe
            index to the selected meta data.

        Args:
            selected_meta: The list of the metadata users want to index with.

        """
        meta_names = list(selected_meta)
        meta_names.append('sample')
        meta_index = []
        for x in meta_names:
            meta_index.append(self.meta.ix[self.data.index][x].values)
        meta_index = np.asarray(meta_index)
        multi_meta_index = pd.MultiIndex.from_arrays(meta_index, names=meta_names)
        self.data.index = multi_meta_index
        # TODO set the index for existing samples in the region dataframe.
        # The index size of the region dataframe does not necessarily be equal to that of metadata df.

    def to_matrix(self, values, selected_regions, default_value=0):
        """Creates a 2D multi-indexed matrix representation of the data.
            This representation allows the data to be sent to the machine learning algorithms.

        Args:
            values: The value or values that are going to fill the matrix.
            selected_regions: The index to one axis of the matrix.
            default_value: The default fill value of the matrix

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

