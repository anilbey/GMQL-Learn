# contains the datamodel class with metadata and region data structures
# provides data manipulation methods
# keeps track of the index
# creates compact views


import pandas as pd
from parser.parser import Parser

# Usage example

class DataModel:
    def __init__(self):
        self.data = None
        return

    def load(self, path, selected_regs, selected_meta, selected_vals):
        p = Parser(path, selected_regs, selected_meta, selected_vals)
        self.data = p.data



# Usage example
# selected_values = 'count_GENES_PATIENTS'
# selected_region_data = ['chr','left','right','strand','gene_id','transcript_id']
# selected_meta_data = ['PATIENTS.manually_curated|sequence_source','PATIENTS.manually_curated|tissue_status','PATIENTS.manually_curated|tumor_description','GENES.description']
# d = DataModel()
# d.load("./whole_data",selected_region_data, selected_meta_data, selected_values)
#
#
#
#
