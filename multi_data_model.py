from data_model import DataModel
from parser.parser import Parser
import pandas as pd
import warnings
import numpy as np

class MultiDataModel:
    """
    Derived DataModel class to represent data that are mapped with multiple references
    
    """

    def __init__(self):
        self.data_model = []
        return

    def load(self, path, genes_uuid, regs=['chr', 'left', 'right', 'strand'], meta=[], values=[], full_load = False):

        if not full_load:
            warnings.warn("\n\n You are using the optimized loading technique. "
                          "All-zero rows are not going to be loaded into memory. "
                          "To load all the data please set the full_load parameter equal to True.")
        p = Parser(path)
        all_meta_data = p.parse_meta(meta)
        all_data = p.parse_data(regs,values, full_load)
        all_data = pd.pivot_table(all_data,
                                   values=values, columns=regs, index=['sample'],
                                   fill_value=0)

        group1 = all_meta_data.groupby([genes_uuid]).count()
        for g in group1.index.values:
            series = all_meta_data[genes_uuid] == g
            m = (all_meta_data[series])
            d = (all_data.loc[series]).dropna(axis=1, how='all')  # not to show the NaN data
            self.data_model.append(DataModel.from_memory(d, m))
            self.all_meta_data = all_meta_data

    def merge(self, samples_uuid):

        all_meta_data = pd.DataFrame()
        for dm in self.data_model:
            all_meta_data = pd.concat([all_meta_data, dm.meta], axis=0)

        group = all_meta_data.groupby([samples_uuid])['sample']
        sample_sets = group.apply(list).values

        merged_df = pd.DataFrame()
        multi_index = list(map(list, zip(*sample_sets)))
        multi_index_names = list(range(0, len(sample_sets[0])))
        i = 1
        for pair in sample_sets:
            i += 1
            numbers = list(range(0, len(pair)))
            df_temp = pd.DataFrame()
            for n in numbers:
                try:  # data.loc[pair[n]] may not be found due to the fast loading (full_load = False)
                    df_temp = pd.concat([df_temp, self.data_model[n].data.loc[pair[n]]], axis=1)
                except:
                    pass
            merged_df = pd.concat([merged_df, df_temp.T.bfill().iloc[[0]]], axis=0)

        multi_index = np.asarray(multi_index)
        multi_index = pd.MultiIndex.from_arrays(multi_index, names=multi_index_names)
        merged_df.index = multi_index
        return merged_df



# sample usage, will be provided in an ipython notebook 
#
# genes_uuid = "GENES.biospecimen_aliquot|bcr_patient_uuid"
# patients_uuid = "PATIENTS.biospecimen_aliquot|bcr_patient_uuid"
# path = "./multi_ref_data/job_multi_ref_anil_20170523_142650_MULTI_REF/files/"
# selected_regs = ['chr', 'left', 'right', 'strand']
# selected_vals = ['count_GENES_PATIENTS']
#
# m = MultiDataModel()
# m.load(path, genes_uuid, selected_regs,[], selected_vals, full_load=True)
# print(m.merge(patients_uuid))
