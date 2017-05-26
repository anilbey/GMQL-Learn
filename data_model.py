import pandas as pd
from parser.parser import Parser
import numpy as np


class DataModel:
    def __init__(self):
        self.data = None
        self.meta = None
        return

    def load(self, _path, regs=['chr', 'left', 'right', 'strand'], meta=[], values=[]):
        """Parses and loads the data into instance attributes.

        Args:
            _path: The path to the dataset on the filesystem.
            regs: the regions that are to be analyzed.
            meta: the meta-data that are to be analyzed.
            values: the values that are to be selected.

        """
        p = Parser(_path)
        self.meta = p.parse_meta(meta)
        self.data = p.parse_data(regs, values)

    @classmethod
    def from_multi_reference(cls):
        pass

    def combine_meta(self, selected_meta):
        """Sets one axis of the 2D multi-indexed dataframe
            index to the selected meta data.

        Args:
            selected_meta: The list of the metadata users want to index with.

        """
        meta_names = list(selected_meta)
        meta_names.append('sample')
        meta_index = []
        selected_meta.append('sample')
        for x in meta_names:
            meta_index.append(self.meta[x].values)
        meta_index = np.asarray(meta_index)
        multi_meta_index = pd.MultiIndex.from_arrays(meta_index, names=selected_meta)
        self.data.index = multi_meta_index

    def to_matrix(self, values, multi_index):
        """Creates a 2D multi-indexed matrix representation of the data.
            This representation allows the data to be sent to the machine learning algorithms.

        Args:
            values: The value or values that are going to fill the matrix.
            multi_index: The index to one axis of the matrix.

        """
        if isinstance(values, list):
            for v in values:
                self.data[v] = self.data[v].map(float)
        else:
            self.data[values] = self.data[values].map(float)
        print("started pivoting")
        self.data = pd.pivot_table(self.data,
                                   values=values, columns=multi_index, index=['sample'],
                                   fill_value=0)
        print("end of pivoting")

