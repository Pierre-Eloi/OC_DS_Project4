#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for data modeling 
with machine learning algorithms.""" 

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Perceptron
from xgboost import XGBRegressor

def linreg_hparams(X, y, estimator, elastic_net=False):
    """ Automated Selection of the hyparameters for linear regression
    The selection is based on grid search.
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        the linear model to be improved
    elastic_net: Bool
        if linear regression with combined L1 and L2 priors as regularizer
    -----------
    Return:
        estimator
    """
    if elastic_net:
        param_grid = [{'alpha': np.logspace(-1, 3, 5),
                       'l1_ratio': np.linspace(0.1, 0.9, 9)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=5,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        p_a = np.log10(grid_search.best_params_['alpha'])
        estimator = grid_search.best_estimator_
        param_grid = [{'alpha': np.logspace(p_a-0.3, p_a+0.7, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
    else:
        param_grid = [{'alpha': np.logspace(-1, 3, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=5,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        p_a = np.log10(grid_search.best_params_['alpha'])
        param_grid = [{'alpha': np.logspace(p_a-0.3, p_a+0.7, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
    model = grid_search.best_estimator_
    return model

 def compare_models(X, y, estimators):
    """ Fonction to compare the RMSE get with the
    most common Machine Learning Regressors.
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimators: list
        List of estimators to be compared
    -----------
    Return:
        DataFrame
    """
    # Train models
    scores = []
    names = []
    std_rmse = []
    for m in models:
        m.fit(X, y)
        m_scores = cross_val_score(m,X, y,
                                   scoring="neg_mean_squared_error",
                                   cv=10)
        m_scores = np.sqrt(-m_scores)
        m_names = type(m).__name__
        scores.append(m_scores.mean())
        std_rmse.append(m_scores.std())
        names.append(m_names)
    # Create the DataFrame
    df = pd.DataFrame({'RMSE_mean': scores, 'RMSE_std': std_rmse}, index=names)
    return df   