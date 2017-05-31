from data_model import DataModel
from parser.parser import Parser
import pandas as pd


class MultiDataModel:
    """
    Derived DataModel class to represent data that are mapped with multiple references
    
    """

    def __init__(self):
        self.data_model = []
        return

    def load(self, path, id_metadata, regs=['chr', 'left', 'right', 'strand'], meta=[], values=[], full_load = False):
        p = Parser(path)
        all_meta_data = p.parse_meta(meta)
        all_data = p.parse_data(regs,values)
        all_data = pd.pivot_table(all_data,
                                   values=values, columns=regs, index=['sample'],
                                   fill_value=0)

        group1 = all_meta_data.groupby([id_metadata]).count()
        for g in group1.index.values:
            series = all_meta_data[id_metadata] == g
            m = (all_meta_data[series])
            d = (all_data.loc[series])
            self.data_model.append(DataModel.from_memory(d, m))
