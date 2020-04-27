#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for
downloading and loading data. 
"""

import os
from six.moves import urllib
import pandas as pd
from IPython.display import display

def download_data(url, data_folder='data'):
    """Download data from an url.
    Parameters
    ----------
    url: string 
        url of the site hosting data
    data_folder: string, default 'data' 
        name of the project folder gathering data  
    -------
    Return
    list object
        List gathering the name of all dowloaded files
    """
    folder_path=os.path.join(data_folder)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        file_name = url.rsplit('/', 1)[-1]
        data_path = os.path.join(folder_path, file_name)
        urllib.request.urlretrieve(url, data_path)
        extension = file_name.rsplit('.', 1)[-1]
        if extension == "zip":
            from zipfile import ZipFile
            with ZipFile(data_path, 'r') as data_zip:
                data_zip.extractall(path=folder_path)
        if extension == "tgz":
            import tarfile
            data_tgz = tarfile.open(data_path)
            data_tgz.extractall(path=folder_path)
    files = []
    # list all extracted files 
    for r, d, f in os.walk(folder_path): # r=root, d=directories, f=files
        for file in f:
            files.append(file)
    print("The extracted files are:")
    for f in files:
        print(f)
    return files

def load_data(csv_file, data_folder='data', sep=','):
    """ Load .csv files into a dataframe.
    Display also the first 3 rows.
        Parameters
    ----------
    csv_file: string
        name of the csv file to be loaded (with .csv extension)
    data_folder: string, default 'data' 
        name of the folder gathering data
    sep : string, default ','
        Delimiter to use    
    -------
    Return
    DataFrame
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.
    """
    csv_path = os.path.join(data_folder, csv_file)
    df = pd.read_csv(csv_path, sep=sep)

    display(df.head(3))
    return df
