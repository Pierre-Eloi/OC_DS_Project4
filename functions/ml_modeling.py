#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for data modeling
with machine learning algorithms."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

def get_models(X, y, best_hparams=False):
    """Get a list of the most common models to evaluate.
    -----------
    Parameters:
    X: Array
        the array object holding data, not used if best_hparams=False
    y: Array
        the target, not used if best_hparams=False
    best_hparams: bool, default False
        if True use specific functions to get best hyperparameters for each model.
        Otherwise use defaut hyperparameters.
    -----------
    Return:
        list object
    """
    if best_hparams:
        ridge = lin_reg_best_params(X, y, Ridge())
        lasso = lin_reg_best_params(X, y, Lasso())
        elastic_net = lin_reg_best_params(X, y, ElasticNet(), elastic_net=True)
        svm_lin_reg = svm_reg_best_params(X, y, LinearSVR())
        svm_rbf_reg = svm_reg_best_params(X, y, SVR(), kernel=True)
        knn_reg = knn_reg_best_params(X, y, KNeighborsRegressor())
        tree_reg = DecisionTreeRegressor()
        forest_reg = RandomForestRegressor()
        gb_reg = GradientBoostingRegressor()
        xgb_reg = XGBRegressor()
        #mlp_reg = MLPRegressor()
    else:
        ridge = Ridge()
        lasso = Lasso()
        elastic_net = ElasticNet()
        svm_lin_reg = LinearSVR(loss='squared_epsilon_insensitive', dual=False)
        svm_rbf_reg = SVR()
        knn_reg = KNeighborsRegressor()
        tree_reg = DecisionTreeRegressor()
        forest_reg = RandomForestRegressor()
        gb_reg = GradientBoostingRegressor()
        xgb_reg = XGBRegressor()
        #mlp_reg = MLPRegressor()
    models = [ridge,
              lasso,
              elastic_net,
              svm_lin_reg,
              svm_rbf_reg,
              knn_reg,
              tree_reg,
              forest_reg,
              #mlp_reg,
              gb_reg,
              xgb_reg]
    return models

def compare_models(X, y, estimators, rfe=False):
    """ Fonction to compare the RMSE and r2 scorers
    for each model of the estimators list.
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimators: list
        List of estimators to be compared
    rfe: bool, default False
        if a feature selection must be carried out
    -----------
    Return:
        DataFrame
    """
    scores_rmse = []
    scores_r2 = []
    names = []
    std_rmse = []
    std_r2 = []
    scoring = {'neg_rmse': 'neg_root_mean_squared_error',
               'r2': 'r2'}
    # Train models
    if rfe:
        list_features = []
        list_masks = []
    for m in estimators:
        if rfe:
            selector = RFECV(estimator=m, cv=cv, scoring=scoring['neg_rmse'])
            X_rfe = selector.fit_transform(X, y)
            list_features.append(selector.n_features_)
            list_masks.append(selector.support_)
            m_scores = cross_validate(m, X_rfe, y, scoring=scoring, cv=cv,
                                      return_train_score=True)
        else:
            m_scores = cross_validate(m, X, y, scoring=scoring, cv=cv,
                                      return_train_score=True)
        m_names = type(m).__name__
        scores_rmse.append(-m_scores['test_neg_rmse'].mean())
        scores_r2.append(m_scores['test_r2'].mean())
        std_rmse.append(-m_scores['test_neg_rmse'].std())
        std_r2.append(m_scores['test_r2'].std())
        names.append(m_names)
    # Create the DataFrame
    df = pd.DataFrame({'RMSE': scores_rmse,
                       'RMSE_std': std_rmse,
                       'R2': scores_r2,
                       'R2_std': std_r2}, index=names)
    if rfe:
        df['N_Features'] = list_features
        df['mask_Features'] = list_masks
    return df

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

def knn_reg_best_params(X, y, estimator, scoring='neg_root_mean_squared_error'):
    """ Automated Selection of the hyparameters for k-NN regression
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
    -----------
    Return:
        estimator
    """
    param_grid = [{'n_neighbors': np.linspace(2, 20, 10, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X, y)
    best_n = grid_search.best_params_['n_neighbors']
    param_grid = [{'n_neighbors': np.linspace(best_n-1, best_n+1, 3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X, y)
    return grid_search.best_estimator_
