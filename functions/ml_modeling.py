#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for data modeling
with machine learning algorithms."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

def lin_reg_best_params(X, y, estimator, scoring='neg_root_mean_squared_error',
                       elastic_net=False):
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
    scoring: str or callable, default 'neg_root_mean_squared_error'
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
        param_grid = [{'alpha': np.logspace(p_a-0.5, p_a+0.5, 11)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
    else:
        param_grid = [{'alpha': np.logspace(-1, 3, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring)
        grid_search.fit(X, y)
        p_a = np.log10(grid_search.best_params_['alpha'])
        param_grid = [{'alpha': np.logspace(p_a-0.5, p_a+0.5, 11)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
    return grid_search.best_estimator_

    from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

def svm_reg_best_params(X, y, estimator, scoring='neg_root_mean_squared_error',
                       kernel=False):
    """ Automated Selection of the hyparameters for SVM regression
    The selection is based on grid search.
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        the SVM model to be improved
    scoring: str or callable, default 'neg_root_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    kernel: bool, default False
        if the kernel trik is used for non_linear data
    -----------
    Return:
        estimator
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if kernel:
        estimator = estimator.set_params(kernel='rbf')
        # Hyperparameter initialization
        param_dict = {'C': np.log10(1),
                      'gamma': round(np.log10(1/(n_features*X.var())), 0),
                      'epsilon': round(np.log10(y.std()*0.2), 0)}
        # Get the right order of magnitude for each hyperparameter
        param_grid = [{'C': np.logspace(param_dict['C']-2, param_dict['C']+2, 5),
                       'gamma': np.logspace(param_dict['gamma']-1, param_dict['gamma']+1, 3),
                       'epsilon': np.logspace(param_dict['epsilon']-1, param_dict['epsilon']+1, 3)}]
        grid_search = GridSearchCV(estimator, param_grid,
                                   cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        estimator = grid_search.best_estimator_
        # Hyperparameter finetuning
        for p in param_dict:
            param_dict[p] = np.log10(grid_search.best_params_[p])
            param_grid = [{p: np.logspace(param_dict[p]-0.5, param_dict[p]+0.5, 11)}]
            grid_search2 = GridSearchCV(estimator, param_grid,
                                        cv=cv, scoring=scoring)
            grid_search2.fit(X, y)
            estimator = grid_search2.best_estimator_
    else:
        if n_samples > n_features:
            dual = False
        else:
            dual = True
        estimator = estimator.set_params(loss='squared_epsilon_insensitive',
                                         dual=dual)
        # Hyperparameter initialization
        param_dict = {'C': np.log10(1),
                      'epsilon': round(np.log10(y.std()*0.2), 0)}
        # Get the right order of magnitude for each hyperparameter
        param_grid = [{'C': np.logspace(param_dict['C']-2, param_dict['C']+2, 5),
                       'epsilon': np.logspace(param_dict['epsilon']-1, param_dict['epsilon']+1, 3)}]
        grid_search = GridSearchCV(estimator, param_grid,
                                   cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        estimator = grid_search.best_estimator_
        # Hyperparameter finetuning
        for p in param_dict:
            param_dict[p] = np.log10(grid_search.best_params_[p])
            param_grid = [{p: np.logspace(param_dict[p]-0.5, param_dict[p]+0.5, 11)}]
            grid_search2 = GridSearchCV(estimator, param_grid,
                                       cv=cv, scoring=scoring)
            grid_search2.fit(X, y)
            estimator = grid_search2.best_estimator_
    return estimator

def compare_models(X, y, estimators, scoring='neg_root_mean_squared_error',
                   rfe=False):
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
    scoring: str or callable, default 'neg_root_mean_squared_error'
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
    if rfe:
        list_features = []
    for m in estimators:
        if rfe:
            selector = RFECV(estimator=m, cv=cv, scoring=scoring)
            X = selector.fit_transform(X, y)
            list_features.append(selector.n_features_)
        m_scores = cross_val_score(m, X, y, scoring=scoring, cv=cv)
        m_names = type(m).__name__
        scores.append(-m_scores.mean())
        std_rmse.append(-m_scores.std())
        names.append(m_names)
    # Create the DataFrame
    df = pd.DataFrame({'RMSE_mean': scores, 'RMSE_std': std_rmse}, index=names)
    if rfe:
        df['N_Features'] = list_features
    return df
