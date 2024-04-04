import os
import pandas as pd


class DataReader:
    @staticmethod
    def read_files(path, **kwargs):
        METADATA_COLS = kwargs.get('metadata_cols', ['Sample', 'Subject', 'Study.Group', 'Age', 'Gender', 'BMI'])

        genera = pd.read_table(f"{path}/genera.tsv")
        metadata = pd.read_table(f"{path}/metadata.tsv")[METADATA_COLS]
        mtb = pd.read_table(f"{path}/mtb.tsv")
        # species = pd.read_table(f"{path}/species.tsv")
        return genera, metadata, mtb

class DataWriter:
    @staticmethod
    def write_files(path, file_name, content):
        extension = file_name.split('.')[-1]
        file_name = file_name.replace(f".{extension}", "")
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(f"{path}/{file_name}"):
            file_name += "_1"
        with open(f"{path}{file_name}.{extension}", "w") as f:
            f.write(content)