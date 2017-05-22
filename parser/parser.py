"""

General purpose parser for the (tab separated) output of the MAP operations of GMQL

"""

import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree


class Parser:
    def __init__(self, path, selected_region_data, selected_meta, selected_values):
        self.path = path
        self.data = None
        self.meta_data = None
        self.schema = self._get_file("schema")
        self.read_all_meta_data(self.path, selected_meta)
        selected_columns = selected_region_data + selected_values
        self.read_all(self.path, selected_columns)

        self.to_matrix(selected_values, selected_region_data)

        meta_index = []
        for x in selected_meta:
            meta_index.append(self.meta_data[x].values)
        meta_index = np.asarray(meta_index)
        multi_meta_index = pd.MultiIndex.from_arrays(meta_index, names=selected_meta)
        self.data.index = multi_meta_index


        return

    def get_sample_id(self, path):
        sp = path.split('/')
        file_name = sp[-1]
        return file_name.split('.')[0]

    def _get_files(self, extension, path):
        # retrieves the files sharing the same extension
        files = []
        for file in os.listdir(path):
            if file.endswith(extension):
                files.append(os.path.join(path, file))
        return sorted(files)

    def _get_file(self, extension):
        for file in os.listdir(self.path):
            if file.endswith(extension):
                return os.path.join(self.path, file)


    def parse_schema(self, schema_file):
        # parses the schema and returns its columns
        e = xml.etree.ElementTree.parse(schema_file)
        root = e.getroot()
        cols = []
        for elem in root.findall(".//{http://genomic.elet.polimi.it/entities}field"):  # XPATH
            cols.append(elem.text)
        return cols

    def _get_sample_name(self, path):
        sp = path.split('/')
        file_name = sp[-1]
        return file_name.split('.')[0]

    def read_meta_data(self, fname, selected_meta_data):
        # reads a meta data file into a dictionary

        columns = []
        data = []
        with open(fname) as f:
            for line in f:
                splitted = line.split('\t')
                columns.append(splitted[0])
                data.append(splitted[1])
        df = pd.DataFrame(data=data, index=columns)
        df = df.T
        sample = self._get_sample_name(fname)
        df = df[selected_meta_data]
        df['sample'] = sample
        return df

    def read_all_meta_data(self, path, selected_meta_data):
        # reads all meta data files
        files = self._get_files("meta", path)
        df = pd.DataFrame()
        for f in files:
            data = self.read_meta_data(f, selected_meta_data)
            if data is not None:
                df = pd.concat([data, df], axis=0)
        self.meta_data = df

    def read_one(self, path, cols, selected_cols):
        # reads a sample file
        df = pd.read_table(path, engine='c', sep="\t", lineterminator="\n")
        df.columns = cols  # column names from schema
        df = df[selected_cols]
        sample = self._get_sample_name(path)
        df['sample'] = sample
        return df

    def select_columns(self, desired_cols):
        self.data = self.data[desired_cols]



    def read_all(self, path, selected_columns):
        # reads all sample files
        files = self._get_files("gdm", path)
        df = pd.DataFrame()
        cols = self.parse_schema(self.schema)
        for f in files:
            data = self.read_one(f, cols, selected_columns)
            if data is not None:
                df = pd.concat([data, df], axis=0)
        self.data = df

    def to_matrix(self, value, multi_index):
        # creates a matrix dataframe
        if isinstance(value, list):
            for v in value:
                self.data[v] = self.data[v].map(float)
        else:
            self.data[value] = self.data[value].map(float) # issue: does not map for multiple values
        self.data = pd.pivot_table(self.data,
                                values=value, columns=multi_index, index=['sample'],
                                fill_value=0)

    def remove_zero_regions(self):
        # to remove the zero regions
        self.data = self.data.loc[(self.data != 0).any(1)]

# Usage example
selected_values = ['count_GENES_PATIENTS']
selected_region_data = ['chr','left','right','strand','gene_id','transcript_id']
selected_meta_data = ['PATIENTS.manually_curated|sequence_source','PATIENTS.manually_curated|tissue_status','PATIENTS.manually_curated|tumor_description','GENES.description']
p = Parser("../mini_data",selected_region_data, selected_meta_data, selected_values)



