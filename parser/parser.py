"""

General purpose parser for the (tab separated) output of the MAP operations of GMQL

"""

import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree


class Parser:
    def __init__(self, path, selected_region_data, selected_meta, selected_values):
        # complete constructor
        # reads
        self.path = path
        self.data = None
        self.meta = None
        self.schema = self._get_file("schema")

        # copy the list since it is to be modified
        regions = list(selected_region_data)
        self.parse_meta(self.path, selected_meta)
        if (type(selected_values) is list):
            regions.extend(selected_values)
        else:
            regions.append(selected_values)
        self.parse_data(self.path, regions, selected_values)
        self.to_matrix(selected_values, selected_region_data)
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

    def parse_single_meta(self, fname, selected_meta_data):
        # reads a meta data file into a dataframe
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
        if selected_meta_data is not None:
            df = df[selected_meta_data]
        df['sample'] = sample
        return df

    def parse_meta(self, path, selected_meta_data):
        # reads all meta data files
        files = self._get_files("meta", path)
        df = pd.DataFrame()
        for f in files:
            data = self.parse_single_meta(f, selected_meta_data)
            if data is not None:
                df = pd.concat([df, data], axis=0)
        df.index = df['sample']
        self.meta = df

    def parse_single_data(self, path, cols, selected_region_data, selected_values):
        # reads a sample file
        df = pd.read_table(path, engine='c', sep="\t", lineterminator="\n")
        df.columns = cols  # column names from schema
        df = df[selected_region_data]
        if (type(selected_values) is list):
            df_2 = pd.DataFrame(dtype=float)
            for value in selected_values:
                df_3 = df.loc[df[value] != 0]
                df_2 = pd.concat([df_2,df_3], axis=0)
            df = df_2
        else:
            df = df.loc[df[selected_values] != 0]
        sample = self._get_sample_name(path)
        df['sample'] = sample
        print(sample,end=' ')
        return df

    def parse_data(self, path, selected_region_data, selected_values):
        # reads all sample files
        files = self._get_files("gdm", path)
        df = pd.DataFrame(dtype=float)

        cols = self.parse_schema(self.schema)
        for f in files:
            data = self.parse_single_data(f, cols, selected_region_data, selected_values)
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
        print("started pivoting")
        self.data = pd.pivot_table(self.data,
                                values=value, columns=multi_index, index=['sample'],
                                fill_value=0)
        print("end of pivoting")