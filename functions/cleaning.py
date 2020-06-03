#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for
selecting and cleaning data.
"""

import numpy as np
import pandas as pd
import ast


def location_handling(data):
    """
    Function to extract all informations from the Location feature.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding the data
    -----------
    Return:
        DataFrame
    """
    df = pd.concat([data.drop('Location', axis=1), # Delete 'Location' column
                    data['Location'].apply(ast.literal_eval) # Convert string format into dictionary one
                                    .apply(pd.Series)], # Split the dictionary into columns
                    axis=1)
    df = pd.concat([df.drop('human_address', axis=1), # Delete 'human_address' column
                    df['human_address'].apply(ast.literal_eval) # Convert string into dictionary
                                       .apply(pd.Series)], # Split the dictionary into columns
                    axis=1)
    return df

def rename_features(data):
    """
    Function to rename features from the 2015 dataset with the same names
    that the ones in the 2016 dataset
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding the data
    -----------
    Return:
        DataFrame
    """
    df = data.rename(columns={'GHGEmissions(MetricTonsCO2e)': 'TotalGHGEmissions',
                              'GHGEmissionsIntensity(kgCO2e/ft2)': 'GHGEmissionsIntensity',
                              'Comment': 'Comments',
                              'latitude': 'Latitude',
                              'longitude': 'Longitude',
                              'address': 'Address',
                              'city': 'City',
                              'state': 'State',
                              'zip': 'ZipCode'})
    return df

def drop_extra_features(data_2015, data_2016):
    """
    Function to drop all features from the 2015 dataset which are not included
    in the 2016 dataset
    -----------
    Parameters:
    data_2015: DataFrame
        the pandas object holding the 2015 data
    data_2016: DataFrame
        the pandas object holding the 2016 data
    -----------
    Return:
        DataFrame
    """
    cols_2015 = data_2015.columns.tolist()
    cols_2016 = data_2016.columns.tolist()
    cols_to_del = [c for c in cols_2015 if c not in cols_2016]
    df = data_2015.drop(columns=cols_to_del)
    return df

def pipe_2015(data_2015, data_2016):
    """
    Function to prepare the 2015 dataset to get the same format that 2016 dataset
    -----------
    Parameters:
    data_2015: DataFrame
        the pandas object holding the 2015 data
    data_2016: DataFrame
        the pandas object holding the 2016 data
    -----------
    Return:
        DataFrame
    """
    df = location_handling(data_2015)
    df = rename_features(df)
    df = drop_extra_features(df, data_2016)
    return df

def handle_targets(data):
    """ Function to discard all no values or outliers for feature targets.
    It also corrects weather normalized outliers.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    # drop missing values
    idx_to_keep = data[['TotalGHGEmissions', 'SiteEnergyUse(kBtu)']].dropna().index
    print("{} buildings will be discarded due to missing data for \
          target features.".format(data.shape[0]-idx_to_keep.size))
    df = data.loc[idx_to_keep]
    # drop outliers
    mask = (df['TotalGHGEmissions']==0)|(df['SiteEnergyUse(kBtu)']==0)
    idx_to_del = df[mask].index
    print("{} outliers will be dropped".format(len(idx_to_del)))
    # correct data where SiteEnergyUseWN(kBtu)=0
    wn_factor = (df['SiteEnergyUseWN(kBtu)']/df['SiteEnergyUse(kBtu)']).mean()
    mask = df['SiteEnergyUseWN(kBtu)']==0
    idx = df[mask].index
    df.loc[idx, 'SiteEnergyUseWN(kBtu)'] = df.loc[idx, 'SiteEnergyUse(kBtu)']*wn_factor
    return df.drop(index=idx_to_del)

def handle_duplicates(data):
    """ Function to check if there are some duplicates in the dataset.
    If any, they will be discarded.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    # check there are no missing values for the primary key OSEBuildingID.
    n_na = data[data['OSEBuildingID'].isna()].size
    if n_na != 0:
        print("OSEBuildingID is not a valid primary key, please use another feature")
        df = data
    else:
        n_dup = data.shape[0] - data['OSEBuildingID'].unique().size
        print("There are {} duplicates".format(n_dup))
        if n_dup !=0:
            df = data.drop_duplicates('OSEBuildingID').set_index('OSEBuildingID')
        else:
            df = data.set_index('OSEBuildingID')
    return df

def handle_outliers(data):
    """ Function to discard outlier by using the Outlier feature.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    idx = data[data['Outlier'].notna()].index
    print("{} buildings will be discarded since considered as outliers.".format(idx.size))
    df = data.drop(index=idx)
    return df.drop(columns='Outlier')

def drop_housing(data):
    """ Function to discard the buildings whose the primary property type is housing.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    list_housing = ['Low-Rise Multifamily', 'Mid-Rise Multifamily', 'High-Rise Multifamily']
    mask = (data['PrimaryPropertyType'].isin(list_housing))|\
           (data['LargestPropertyUseType']=='Multifamily Housing')
    idx_to_del = data[mask].index
    print("{} buildings dedicated mainly to housing will be discarded".format(len(idx_to_del)))
    df = data.drop(index=idx_to_del)
    return df

def energy_conversion(data):
    """Function to convert the energy consumption features which are not in kBtu.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    # Create a copy of the dataset
    df = data.copy()
    # Convert the Electricity feature
    df['Electricity(kBtu)'] = df['Electricity(kWh)'] * 3.41214
    # Convert the Natural Gas feature
    df['NaturalGas(kBtu)'] = df['NaturalGas(therms)'] * 100
    # Drop redundant features (features not in kBtu)
    col_to_del = ['Electricity(kWh)', 'NaturalGas(therms)']
    df.drop(columns=col_to_del, inplace=True)
    return df

def conso_outliers(data):
    """ Function to discard the obvious outliers
    for the features linked to the energy consumption,
    no matter the source of energy.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    col_conso = ['SiteEnergyUse(kBtu)',
                 'SiteEnergyUseWN(kBtu)',
                 'SteamUse(kBtu)',
                 'Electricity(kBtu)',
                 'NaturalGas(kBtu)']
    # Since they are consumptions, all data must be positive
    mask = (data[col_conso]>=0).product(axis=1)
    idx_to_del = data[mask==0].index
    df = data.drop(index=idx_to_del)
    # Each consumption must be inferior to the total consumption
    mask = (data['SteamUse(kBtu)']>data['SiteEnergyUse(kBtu)'])|\
           (data['Electricity(kBtu)']>data['SiteEnergyUse(kBtu)'])|\
           (data['NaturalGas(kBtu)']>data['SiteEnergyUse(kBtu)'])
    idx_to_del = data[mask].index
    print("{} buildings will be discarded due to anomalous consumptions".format(len(idx_to_del)))
    df = data.drop(index=idx_to_del)
    return df

def pipe_cleaning(data, conso_del=False):
    """Pipeline to clean the dataset.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    conso_del: bool, default False
        to discard or not the obvious outliers for the features linked to the energy consumption
    -----------
    Return:
        DataFrame
    """
    df = handle_targets(data)
    df = handle_duplicates(df)
    df = handle_outliers(df)
    df = drop_housing(df)
    df = energy_conversion(df)
    if conso_del:
        df = conso_outliers(df)
    return df
