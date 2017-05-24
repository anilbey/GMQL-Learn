# contains the datamodel class with metadata and region data structures
# provides data manipulation methods
# keeps track of the index
# creates compact views


import pandas as pd
from parser.parser import Parser
import numpy as np


class DataModel:
    def __init__(self):
        self.data = None
        self.meta = None
        return

    def load(self, path, regs=[], meta=[], values=[]):
        p = Parser(path, regs, meta, values)
        self.data = p.data
        self.meta = p.meta

    def combine_meta(self, selected_meta):
        meta_names = list(selected_meta)
        meta_names.append('sample')
        meta_index = []
        selected_meta.append('sample')
        for x in meta_names:
            meta_index.append(self.meta[x].values)
        meta_index = np.asarray(meta_index)
        multi_meta_index = pd.MultiIndex.from_arrays(meta_index, names=selected_meta)
        self.data.index = multi_meta_index

