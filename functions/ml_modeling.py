#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for data modeling
with machine learning algorithms."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

def linreg_best_params(X, y, estimator, scoring='neg_mean_squared_error', elastic_net=False):
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
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    elastic_net: bool, default False
        if linear regression with combined L1 and L2 priors as regularizer
    -----------
    Return:
        estimator
    """
    if elastic_net:
        param_grid = [{'alpha': np.logspace(-1, 3, 5),
                       'l1_ratio': np.linspace(0.1, 0.9, 9)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        p_a = np.log10(grid_search.best_params_['alpha'])
        estimator = grid_search.best_estimator_
        param_grid = [{'alpha': np.logspace(p_a-0.3, p_a+0.7, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
    else:
        param_grid = [{'alpha': np.logspace(-1, 3, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring)
        grid_search.fit(X, y)
        p_a = np.log10(grid_search.best_params_['alpha'])
        param_grid = [{'alpha': np.logspace(p_a-0.3, p_a+0.7, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
    model = grid_search.best_estimator_
    return model

def compare_models(X, y, estimators, scoring='neg_mean_squared_error', rfe=False):
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
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    rfe: bool, default False
        if a feature selection must be carried out
    -----------
    Return:
        DataFrame
    """
    # Train models
    scores = []
    names = []
    std_rmse = []
    for m in estimators:
        if rfe:
            selector = RFECV(estimator=m, cv=cv, scoring=scoring)
            selector.fit(X, y)
            n_features = selector.n_features_
            scores = selector.grid_scores_
            m_scores = selector.grid_scores_[n_features-1]
        else:
            m_scores = cross_val_score(m, X, y,
                                       scoring=scoring,
                                       cv=cv)
        m_scores = np.sqrt(-m_scores)
        m_names = type(m).__name__
        scores.append(m_scores.mean())
        std_rmse.append(m_scores.std())
        names.append(m_names)
    # Create the DataFrame
    df = pd.DataFrame({'RMSE_mean': scores, 'RMSE_std': std_rmse}, index=names)
    return df
