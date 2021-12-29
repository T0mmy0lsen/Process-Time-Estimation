from os import listdir
from os.path import isfile, join
import pandas as pd


# Takes all files in a folder that contains a given string
# Returns a list of files
def get_files_containing(path, contains):
    return [f for f in listdir(path) if isfile(join(path, f)) and contains in f]


# Sets the obj.df to a dataframe given a input_path is a csv.
def load_files_to_dataframe(obj):
    lst = get_files_containing(obj.input_path(), obj.name())
    for el in lst:
        if obj.df() is None:
            obj._df = pd.read_csv(obj.input_path() + el, encoding='utf-16', sep=';')
        else:
            obj._df = pd.concat([obj.df(), pd.read_excel(obj.input_path() + el)], axis=0, ignore_index=True)
