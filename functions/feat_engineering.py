#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for feature engineering.""" 

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def add_target_log(data):
    """Function to add the log of the targets.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
        a list with all target
    """
    targets = ['TotalGHGEmissions',
               'SiteEnergyUse(kBtu)',
               'SiteEnergyUseWN(kBtu)']
    df = data.copy()
    targets_log = []
    for c in targets:
        name = c + "_log"
        df[name] = np.log(df[c])
        targets_log.append(name)
    targets += targets_log
    return df, targets

def impute_nan(data):
    """Function impute no values.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    df = data.copy()
    # impute 'SecondLargestPropertyUseTypeGFA' & 'ThirdLargestPropertyUseTypeGFA'
    values={'SecondLargestPropertyUseTypeGFA': 0,
            'ThirdLargestPropertyUseTypeGFA': 0}
    df.fillna(values, inplace=True)
    # impute 'LargestPropertyUseType'
    df['LargestPropertyUseType'].fillna(df['PrimaryPropertyType'], inplace=True)
    # impute 'LargestPropertyUseTypeGFA'
    df['LargestPropertyUseTypeGFA'].fillna(df['PropertyGFATotal'], inplace=True)
    # impute 'NumberofFloors' with the median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df['NumberofFloors'].values.reshape(-1, 1))
    df['NumberofFloors'] = X
    return df

def add_conso_ratio(data):
    """Function to add the proportion of each energy used.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    att_conso = ['SteamUse(kBtu)',
                 'Electricity(kBtu)',
                 'NaturalGas(kBtu)']
    df = data.copy()
    # Add the "OtherFuel(kBtu)" feature
    df['OtherFuel(kBtu)'] = df['SiteEnergyUse(kBtu)'] - df[att_conso].sum(axis=1)
    # Equalize all negative values to 0
    cond = df['OtherFuel(kBtu)'] >= 0
    df['OtherFuel(kBtu)'].where(cond, 0, inplace=True)
    att_conso += ['OtherFuel(kBtu)']
    # Add proportions
    for c in att_conso:
        name = c.replace("(kBtu)", "") + "_ratio"
        df[name] = df[c] / df[att_conso].sum(axis=1)
    return df

def add_gfa_ratio(data):
    """Function to add the proportion of the area assigned to Buildings and the one assigned to Parkings.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    att_gfa = ['PropertyGFAParking',
               'PropertyGFABuilding(s)']
    df = data.copy()
    for c in att_gfa:
        name = c + "_ratio"
        df[name] = df[c] / df[att_gfa].sum(axis=1)
    return df

def add_associative_array(data):
    """Function to create an associative array to standardize the building use categories.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    missing_cat = ['Other - Restaurant/Bar',
                   'Vocational School',
                   'Swimming Pool',
                   'Convenience Store without Gas Station',
                   'Multifamily Housing',
                   'Bar/Nightclub',
                   'Food Sales',
                   'Fast Food Restaurant',
                   'Enclosed Mall']
    cat_to_be_added = ['Restaurant/Bar',
                       'Other',
                       'Other',
                       'Retail Store',
                       'Multifamily Housing',
                       'Restaurant/Bar',
                       'Retail Store',
                       'Restaurant/Bar',
                       'Supermarket/Grocery Store']
    df = data.copy()
    # Drop the \n string at the end of a couple of PrimaryPropertyType modalities
    df['PrimaryPropertyType'] = df['PrimaryPropertyType'].str.replace("\n", "")
    # Replace "Restaurant" by "Restaurant/Bar"
    df['PrimaryPropertyType'] = df['PrimaryPropertyType'].str.replace("Restaurant", "Restaurant/Bar")
    # Gather "Small- and Mid-Sized Office" and "Large Office" modalities in a unique "Office" modality.
    offices = ['Small- and Mid-Sized Office', 'Large Office']
    idx = df[df['PrimaryPropertyType'].isin(offices)].index
    df.loc[idx, 'PrimaryPropertyType'] = "Office"
    # Create the associative array
    associative_df = df[['LargestPropertyUseType', 'PrimaryPropertyType']].dropna()
    associative_df = (associative_df.rename(columns={'LargestPropertyUseType': 'PropertyUseType'})
                                .drop_duplicates()
                                .reset_index(drop=True))
    # Drop the "Mixed Use Property" category
    idx = associative_df[associative_df['PrimaryPropertyType']=="Mixed Use Property"].index
    associative_df.drop(index=idx, inplace=True)
    # Add unreferenced memories
    extra_df = pd.DataFrame({'PropertyUseType': missing_cat,
                             'PrimaryPropertyType': cat_to_be_added})
    associative_df = associative_df.append(extra_df, ignore_index=True)
    return associative_df

def std_use_type(data, associative_df):
    """Function to standardize Use Type categories.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    associative_df: DataFrame
        the pandas object holding the associative array   
    -----------
    Return:
        DataFrame
    """
    cols_UseType = ['LargestPropertyUseType',
                    'SecondLargestPropertyUseType',
                    'ThirdLargestPropertyUseType']
    df = data.copy()
    for c in cols_UseType:
        df[c] = (pd.merge(df[c], associative_df, how='left', left_on=c, right_on='PropertyUseType')
                   .PrimaryPropertyType
                   .tolist())
    return df

def add_use_type_ratio(data):
    """Function to add the gfa proportion for
    the first three largest use type.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    use_gfa = ['LargestPropertyUseTypeGFA',
               'SecondLargestPropertyUseTypeGFA',
               'ThirdLargestPropertyUseTypeGFA']
    df = data.copy()
    for c in use_gfa:
        name = c + "_ratio"
        df[name] = df[c] / df[use_gfa].sum(axis=1)
    return df

def encoding_use_type(data, associative_df):
    """Function to add a feature for each use type category.
    Each feature will be populated with the builfing gfa proprotion
    dedicated to this feature.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    associative_df: DataFrame
        the pandas object holding the associative array     
    -----------
    Return:
        DataFrame
    """
    cols_UseType = ['LargestPropertyUseType',
                    'SecondLargestPropertyUseType',
                    'ThirdLargestPropertyUseType']
    use_gfa_ratio = ['LargestPropertyUseTypeGFA_ratio',
                     'SecondLargestPropertyUseTypeGFA_ratio',
                     'ThirdLargestPropertyUseTypeGFA_ratio']
    df = data.copy()
    list_cat = associative_df['PrimaryPropertyType'].unique().tolist()
    for c in list_cat:
        df[c] = (df[cols_UseType[0]]==c)*df[use_gfa_ratio[0]] + \
                (df[cols_UseType[1]]==c)*df[use_gfa_ratio[1]] + \
                (df[cols_UseType[2]]==c)*df[use_gfa_ratio[2]]
    return df

def OHE_neiborhood(data):
    """Function to one hot encoding the neiborhood feature.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
        int, number of the neiborhood modalities
    """
    df = data.copy()
    cat_encoder = OneHotEncoder()
    neighborhood_1hot = cat_encoder.fit_transform(df['Neighborhood'].values.reshape(-1, 1))
    neighborhood_cat = cat_encoder.categories_[0].tolist()
    df_1hot = pd.DataFrame(neighborhood_1hot.toarray(),
                           columns=neighborhood_cat,
                           index=df.index)
    df = pd.concat([df, df_1hot], axis=1)
    n = len(neighborhood_cat)
    return df, n

def select_feature(data):
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
                   'ZipCode',
                   'SteamUse(kBtu)',
                   'Electricity(kBtu)',
                   'NaturalGas(kBtu)',
                   'OtherFuel(kBtu)',
                   'PropertyGFAParking',
                   'PropertyGFABuilding(s)',
                   'PrimaryPropertyType',
                   'ListOfAllPropertyUseTypes',
                   'LargestPropertyUseType',
                   'SecondLargestPropertyUseType',
                   'ThirdLargestPropertyUseType',
                   'LargestPropertyUseTypeGFA',
                   'SecondLargestPropertyUseTypeGFA',
                   'ThirdLargestPropertyUseTypeGFA',
                   'Neighborhood']
    df = data.drop(columns=cols_to_del)
    return df.set_index('OSEBuildingID')

def sort_columns(data, targets):
    """Function to put target at the end.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    df = data.copy()
    cols = [c for c in df.columns if c not in targets]
    cols += targets
    return df[cols]

def scale_data(data, n, targets):
    """Function to Standardize features by removing the mean and
    scaling to unit variance.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    df = data.copy()
    n += len(targets)
    X = data.iloc[:, :-n].values
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X)
    df.iloc[:, :-n] = X_scaled
    return df

def pipe_engineering(data):
    """Feature engineering pipeline.
    -----------
    Parameters:
    data: DataFrame
        the pandas object holding data
    -----------
    Return:
        DataFrame
    """
    df, targets = add_target_log(data)
    df = impute_nan(df)
    df = add_conso_ratio(df)
    df = add_gfa_ratio(df)
    asso_df = add_associative_array(data)
    df = std_use_type(df, asso_df)
    df = add_use_type_ratio(df)
    df = encoding_use_type(df, asso_df)
    df, n = OHE_neiborhood(df)
    df = select_feature(df)
    df = sort_columns(df, targets)
    df = scale_data(df, n, targets)
    return df