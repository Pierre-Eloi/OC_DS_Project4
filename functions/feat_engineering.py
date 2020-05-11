#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for feature engineering.""" 

import numpy as np
import pandas as pd

def select_feature():
    """Function to drop irrelevant features.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    cols_to_del = ['DataYear',
               'ComplianceStatus',
               'City',
               'State',
               'PropertyName',
               'TaxParcelIdentificationNumber',
               'Address',
               'YearsENERGYSTARCertified',
               'DefaultData',
               'Comments',
               'BuildingType',
               'CouncilDistrictCode',
               'ZipCode']
    df = data.drop(columns=cols_to_del)
    return df.set_index('OSEBuildingID')

def target_log():
    """Function to add the natural logarithm of the targets.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data   
    -----------
    Return:
        DataFrame
    """
    df = data_2015.copy()
    df['SiteEnergyUse(kBtu)_log'] = np.log(df['SiteEnergyUse(kBtu)'])
    df['TotalGHGEmissions_log'] = np.log(df['TotalGHGEmissions'])
    return df