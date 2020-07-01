#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for data modeling
with machine learning algorithms."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
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
        tree_reg = tree_reg_best_params(X, y, DecisionTreeRegressor())
        forest_reg = forest_reg_best_params(X, y, RandomForestRegressor())
        gboost_reg = GradientBoostingRegressor()
        xgboost_reg = XGBRegressor()
        mlp_reg = MLPRegressor()
    else:
        ridge = Ridge()
        lasso = Lasso()
        elastic_net = ElasticNet()
        svm_lin_reg = LinearSVR(loss='squared_epsilon_insensitive', dual=False)
        svm_rbf_reg = SVR()
        knn_reg = KNeighborsRegressor()
        tree_reg = DecisionTreeRegressor(random_state=42)
        forest_reg = RandomForestRegressor(random_state=42)
        gboost_reg = GradientBoostingRegressor(random_state=42)
        xgboost_reg = XGBRegressor()
        mlp_reg = MLPRegressor()
    models = [ridge,
              lasso,
              elastic_net,
              svm_lin_reg,
              svm_rbf_reg,
              knn_reg,
              tree_reg,
              forest_reg,
              gboost_reg,
              xgboost_reg,
              mlp_reg]
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
    scoring = {'neg_mse': 'neg_mean_squared_error',
               'r2': 'r2'}
    # Train models
    if rfe:
        list_features = []
        list_masks = []
    for m in estimators:
        if rfe:
            selector = RFECV(estimator=m, cv=cv, scoring=scoring['neg_mse'],
                             n_jobs=-1)
            X_rfe = selector.fit_transform(X, y)
            list_features.append(selector.n_features_)
            list_masks.append(selector.support_)
            m_scores = cross_validate(m, X_rfe, y, scoring=scoring, cv=cv,
                                      return_train_score=True, n_jobs=-1)
        else:
            m_scores = cross_validate(m, X, y, scoring=scoring, cv=cv,
                                      return_train_score=True, n_jobs=-1)
        m_names = type(m).__name__
        scores_rmse.append(np.sqrt(-m_scores['test_neg_mse']).mean())
        scores_r2.append(m_scores['test_r2'].mean())
        std_rmse.append(np.sqrt(-m_scores['test_neg_mse']).std())
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

def select_features(X, y, estimator, scoring='neg_mean_squared_error'):
    """ Automated Selection of the most significant features.
    The importance weight of each feature is computed for a Random Forest Model.
    All features with an importance weight above a given threshold are selected.
    A grid search is used to find the optimum threshold.
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        Must be a Random Forest model
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    -----------
    Return:
        the mask of the features selected
    """
    n_features = X.shape[1]
    thresholds = np.sort(estimator.feature_importances_)
    selector = SelectFromModel(estimator=estimator)
    pipeline = Pipeline([('selector', selector),
                         ('model', estimator)])                     
    # First search with a randomized one
    n_iter = min([20, int(n_features/5)])
    param_grid = [{'selector__threshold': [thresholds[i] for i in range(0, n_features, 5)]}]
    rand_search = RandomizedSearchCV(pipeline, param_grid, cv=cv, n_iter=n_iter,
                                     scoring=scoring, random_state=42, n_jobs=-1)
    rand_search.fit(X, y)
    # Second search with a grid one
    threshold = rand_search.best_params_['selector__threshold']
    idx = np.where(thresholds==threshold)[0][0]
    if idx==0:
        param_grid = [{'selector__threshold': [thresholds[i] for i in range(idx, idx+5)]}]
    else:
        param_grid = [{'selector__threshold': [thresholds[i] for i in range(idx-4, idx+5)]}]
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_['selector'].get_support()

def lin_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error',
                       elastic_net=False):
    """ Automated Selection of the hyparameters for linear regression
    The selection is based on three successive grid searches.
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
        # First grid search
        param_grid = [{'alpha': np.logspace(-1, 3, 5),
                       'l1_ratio': np.linspace(0.05, 0.95, 19)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        alpha = grid_search.best_params_['alpha']
        estimator = grid_search.best_estimator_
        # Second grid search
        param_grid = [{'alpha': np.linspace(alpha/2, alpha*5, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring,  n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search
        alpha= grid_search.best_params_['alpha']
        param_grid = [{'alpha': np.linspace(alpha*5/6, alpha*7/6, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring,  n_jobs=-1)
        grid_search.fit(X, y)

    else:
        # First grid search
        param_grid = [{'alpha': np.logspace(-1, 3, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        alpha = grid_search.best_params_['alpha']
        # Second grid search
        param_grid = [{'alpha': np.linspace(alpha/2, alpha*5, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring,  n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search
        alpha= grid_search.best_params_['alpha']
        param_grid = [{'alpha': np.linspace(alpha*5/6, alpha*7/6, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring,  n_jobs=-1)
        grid_search.fit(X, y)
    return grid_search.best_estimator_

def svm_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error',
                       kernel=False):
    """ Automated Selection of the hyparameters for SVM regression
    The selection is based on three successive searches.
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        the SVM model to be improved
    scoring: str or callable, default 'neg_mean_squared_error'
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
        c = np.log10(1)
        gamma = round(np.log10(1/(n_features*X.var())), 0)
        epsilon = round(np.log10(y.std()*0.2), 0)
        # First search to Get the right order of magnitude for each hyperparameter
        param_grid = [{'C': np.logspace(c-2, c+2, 5),
                       'gamma': np.logspace(gamma-1, gamma+1, 3),
                       'epsilon': np.logspace(epsilon-1, epsilon+1, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Second search with a randomized search
        c = grid_search.best_params_['C']
        gamma = grid_search.best_params_['gamma']
        epsilon = grid_search.best_params_['epsilon']
        param_grid = [{'C': np.linspace(c, c*5, 10),
                       'gamma': np.linspace(gamma, gamma*5, 10),
                       'epsilon': np.linspace(epsilon, epsilon*5, 10)}]
        rand_search = RandomizedSearchCV(estimator, param_grid, cv=cv, n_iter=100,
                                         scoring=scoring, n_jobs=-1, random_state=42)
        rand_search.fit(X, y)
        # Third search with a grid search
        c = rand_search.best_params_['C']
        gamma = rand_search.best_params_['gamma']
        epsilon = rand_search.best_params_['epsilon']
        param_grid = [{'C': np.linspace(c*5/6, c*7/6, 3),
                       'gamma': np.linspace(gamma*5/6, gamma*7/6, 3),
                       'epsilon': np.linspace(epsilon*7/6, epsilon*7/6, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
    else: # if linear SVM
        if n_samples > n_features:
            dual = False
        else:
            dual = True
        estimator = estimator.set_params(loss='squared_epsilon_insensitive',
                                         dual=dual)
        # Hyperparameter initialization
        c = np.log10(1)
        gamma = round(np.log10(1/(n_features*X.var())), 0)
        epsilon = round(np.log10(y.std()*0.2), 0)
        # First grid search to Get the right order of magnitude for each hyperparameter
        param_grid = [{'C': np.logspace(c-2, c+2, 5),
                       'epsilon': np.logspace(epsilon-1, epsilon+1, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Second grid search
        c = grid_search.best_params_['C']
        epsilon = grid_search.best_params_['epsilon']
        param_grid = [{'C': np.linspace(c/2, c*5, 10),
                       'epsilon': np.linspace(epsilon/2, epsilon*5, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                         scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search
        c = grid_search.best_params_['C']
        epsilon = grid_search.best_params_['epsilon']
        param_grid = [{'C': np.linspace(c*5/6, c*7/6, 3),
                       'epsilon': np.linspace(epsilon*5/6, epsilon*7/6, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
    return grid_search.best_estimator_

def knn_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error'):
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
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    -----------
    Return:
        estimator
    """
    param_grid = [{'n_neighbors': np.linspace(2, 20, 10, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    best_n = grid_search.best_params_['n_neighbors']
    param_grid = [{'n_neighbors': np.linspace(best_n-1, best_n+1, 3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def tree_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error'):
    """ Automated Selection of the hyparameters for Decision Tree Regressor
    The selection is based on three successive grid searches.
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        the Decision Tree model to be improved
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    -----------
    Return:
        estimator
    """
    estimator = estimator.set_params(random_state=42)
    n_samples = X.shape[0]
    # First grid search to get the right order of magnitude for each hyperparameter
    param_grid = [{'min_samples_split': np.logspace(1, 3, 3, dtype=int),
                   'max_features': ['auto', 'sqrt', 'log2'],
                   'max_leaf_nodes': np.logspace(1, int(np.log10(n_samples))-1,
                                                 int(np.log10(n_samples))-1,
                                                 dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator = grid_search.best_estimator_
    # Second grid search
    min_samples_split = grid_search.best_params_['min_samples_split']
    max_leaf = grid_search.best_params_['max_leaf_nodes']
    param_grid = [{'min_samples_split': np.linspace(min_samples_split/2,
                                                   min_samples_split*5,
                                                    10, dtype=int),
                   'max_leaf_nodes': np.linspace(max_leaf/2,
                                                 max_leaf*5,
                                                 10, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    # Third grid search
    min_samples_split = grid_search.best_params_['min_samples_split']
    max_leaf = grid_search.best_params_['max_leaf_nodes']
    param_grid = [{'min_samples_split': np.linspace(min_samples_split/2,
                                                   min_samples_split*3/2,
                                                   3, dtype=int),
                   'max_leaf_nodes': np.linspace(max_leaf*5/6,
                                                 max_leaf*7/6,
                                                 3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def forest_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error'):
    """ Automated Selection of the hyparameters for Random Forest Regressor
    The selection is based three successive searches.
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        the Random Forest model to be improved
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    -----------
    Return:
        estimator
    """
    estimator = estimator.set_params(random_state=42,
                                     n_jobs=-1)
    # First search with a grid one
    param_grid = [{'n_estimators': [100, 500],
                   'max_features': np.linspace(0.2, 0.8, 4)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                     scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    # Second search with a grid one
    n_estimators = grid_search.best_params_['n_estimators']
    max_features = grid_search.best_params_['max_features']
    param_grid = [{'n_estimators': np.linspace(n_estimators/2, n_estimators*3/2,
                                               3, dtype=int),
                   'max_features': np.linspace(max_features-0.1, max_features+0.1, 3)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                     scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    # Second search with a grid one
    n_estimators = grid_search.best_params_['n_estimators']
    max_features = grid_search.best_params_['max_features']
    param_grid = [{'n_estimators': np.linspace(n_estimators*5/6, n_estimators*7/6,
                                               3, dtype=int),
                   'max_features': np.linspace(max_features-0.05, max_features+0.05, 3)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


