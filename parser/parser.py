"""

General purpose parser for the (tab separated) output of the MAP operations of GMQL

"""

import pandas as pd
import os
import xml.etree.ElementTree


class Parser:

    def __init__(self, path):
        self.path = path
        self.schema = self._get_file("schema", path)
        return

    @staticmethod
    def get_sample_id(path):
        sp = path.split('/')
        file_name = sp[-1]
        return file_name.split('.')[0]

    @staticmethod
    def _get_files(extension, path):
        # retrieves the files sharing the same extension
        files = []
        for file in os.listdir(path):
            if file.endswith(extension):
                files.append(os.path.join(path, file))
        return sorted(files)

    @staticmethod
    def _get_file(extension, path):
        for file in os.listdir(path):
            if file.endswith(extension):
                return os.path.join(path, file)

    @staticmethod
    def parse_schema(schema_file):
        # parses the schema and returns its columns
        e = xml.etree.ElementTree.parse(schema_file)
        root = e.getroot()
        cols = []
        for elem in root.findall(".//{http://genomic.elet.polimi.it/entities}field"):  # XPATH
            cols.append(elem.text)
        return cols

    @staticmethod
    def _get_sample_name(path):
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
                data.append(splitted[1].split('\n')[0])  # to remove the \n values
        df = pd.DataFrame(data=data, index=columns)
        df = df.T
        sample = self._get_sample_name(fname)
        if selected_meta_data:  # if null then keep all the columns
            df = df[selected_meta_data]
        df['sample'] = sample
        return df

    def parse_meta(self, selected_meta_data):
        # reads all meta data files
        files = self._get_files("meta", self.path)
        df = pd.DataFrame()
        for f in files:
            data = self.parse_single_meta(f, selected_meta_data)
            if data is not None:
                df = pd.concat([df, data], axis=0)
        df.index = df['sample']
        return df

    def parse_single_data(self, path, cols, selected_region_data, selected_values, full_load):
        # reads a sample file
        df = pd.read_table(path, engine='c', sep="\t", lineterminator="\n")
        df.columns = cols  # column names from schema
        df = df[selected_region_data]

        if not full_load:
            if type(selected_values) is list:
                df_2 = pd.DataFrame(dtype=float)
                for value in selected_values:
                    df_3 = df.loc[df[value] != 0]
                    df_2 = pd.concat([df_2, df_3], axis=0)
                df = df_2
            else:
                df = df.loc[df[selected_values] != 0]

        sample = self._get_sample_name(path)
        df['sample'] = sample
        return df

    def parse_data(self, selected_region_data, selected_values, full_load=False):
        # reads all sample files
        regions = list(selected_region_data)
        if type(selected_values) is list:
            regions.extend(selected_values)
        else:
            regions.append(selected_values)

        files = self._get_files("gdm", self.path)
        df = pd.DataFrame(dtype=float)

        cols = self.parse_schema(self.schema)
        for f in files:
            data = self.parse_single_data(f, cols, regions, selected_values, full_load)
            df = pd.concat([data, df], axis=0)
        return df

