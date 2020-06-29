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
        gb_reg = GradientBoostingRegressor()
        xgb_reg = XGBRegressor()
        mlp_reg = MLPRegressor()
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
        mlp_reg = MLPRegressor()
    models = [ridge,
              lasso,
              elastic_net,
              svm_lin_reg,
              svm_rbf_reg,
              knn_reg,
              tree_reg,
              forest_reg,
              mlp_reg,
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
    threshold = int(np.log10(1 / n_features))
    threshold_min = int(np.log10(estimator.fit(X, y).feature_importances_.min()))-1
    n_threshold = threshold - threshold_min + 1
    selector = SelectFromModel(estimator=estimator)
    pipeline = Pipeline([('selector', selector),
                         ('model', estimator)])
    # Get the right order of magnitude
    param_grid = [{'selector__threshold': np.logspace(threshold_min, threshold, n_threshold)}]
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    threshold = np.log10(grid_search.best_params_['selector__threshold'])
    if threshold==threshold_min:
        return grid_search.best_estimator_['selector'].get_support()
    else:
        # Second grid search
        param_grid = [{'selector__threshold': np.logspace(threshold-0.3, threshold+0.7, 5)}]
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search to finetune hyperparameters
        threshold = np.log10(grid_search.best_params_['selector__threshold'])
        param_grid = [{'selector__threshold': np.logspace(threshold-0.1, threshold+0.1, 3)}]
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_['selector'].get_support()

def lin_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error',
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
                       'l1_ratio': np.linspace(0.1, 0.9, 9)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        alpha = np.log10(grid_search.best_params_['alpha'])
        estimator = grid_search.best_estimator_
        # Second grid search
        param_grid = [{'alpha': np.logspace(alpha-0.3, alpha+0.7, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring,  n_jobs=-1)
        grid_search.fit(X, y)
        alpha= np.log10(grid_search.best_params_['alpha'])
        # Third grid search
        param_grid = [{'alpha': np.logspace(alpha-0.1, alpha+0.1, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring,  n_jobs=-1)
        grid_search.fit(X, y)

    else:
        # First grid search
        param_grid = [{'alpha': np.logspace(-1, 3, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        alpha = np.log10(grid_search.best_params_['alpha'])
        # Second grid search
        param_grid = [{'alpha': np.logspace(alpha-0.3, alpha+0.7, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring,  n_jobs=-1)
        grid_search.fit(X, y)
        alpha= np.log10(grid_search.best_params_['alpha'])
        # Third grid search
        param_grid = [{'alpha': np.logspace(alpha-0.1, alpha+0.1, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring,  n_jobs=-1)
        grid_search.fit(X, y)
    return grid_search.best_estimator_

    from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

def svm_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error',
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
        # Get the right order of magnitude for each hyperparameter
        param_grid = [{'C': np.logspace(c-2, c+2, 5),
                       'gamma': np.logspace(gamma-1, gamma+1, 3),
                       'epsilon': np.logspace(epsilon-1, epsilon+1, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Second grid search
        c = np.log10(grid_search.best_params_['C'])
        gamma = np.log10(grid_search.best_params_['gamma'])
        epsilon = np.log10(grid_search.best_params_['epsilon'])
        param_grid = [{'C': np.logspace(c-0.3, c+0.7, 5),
                       'gamma': np.logspace(gamma-0.3, gamma+0.7, 5),
                       'epsilon': np.logspace(epsilon-0.3, epsilon+0.7, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search to finetune hyperparameters
        c = np.log10(grid_search.best_params_['C'])
        gamma = np.log10(grid_search.best_params_['gamma'])
        epsilon = np.log10(grid_search.best_params_['epsilon'])
        param_grid = [{'C': np.logspace(c-0.1, c+0.1, 3),
                       'gamma': np.logspace(gamma-0.1, gamma+0.1, 3),
                       'epsilon': np.logspace(epsilon-0.1, epsilon+0.1, 3)}]
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
        # Get the right order of magnitude for each hyperparameter
        param_grid = [{'C': np.logspace(c-2, c+2, 5),
                       'epsilon': np.logspace(epsilon-1, epsilon+1, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Second grid search
        c = np.log10(grid_search.best_params_['C'])
        epsilon = np.log10(grid_search.best_params_['epsilon'])
        param_grid = [{'C': np.logspace(c-0.3, c+0.7, 5),
                       'epsilon': np.logspace(epsilon-0.3, epsilon+0.7, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                                   scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search to finetune hyperparameters
        c = np.log10(grid_search.best_params_['C'])
        epsilon = np.log10(grid_search.best_params_['epsilon'])
        param_grid = [{'C': np.logspace(c-0.1, c+0.1, 3),
                       'epsilon': np.logspace(epsilon-0.1, epsilon+0.1, 3)}]
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
    The selection is based on grid search.
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
    # Get the right order of magnitude for each hyperparameter
    param_grid = [{'min_samples_split': np.logspace(0, 2, 3, dtype=int),
                   'min_samples_leaf': np.logspace(0, 2, 3, dtype=int),
                   'max_features': ['auto', 'sqrt', 'log2'],
                   'max_leaf_nodes': np.logspace(1, int(np.log10(n_samples))-1,
                                                 int(np.log10(n_samples))-1,
                                                 dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator = grid_search.best_estimator_
    # Second grid search
    min_samples_split = np.log10(grid_search.best_params_['min_samples_split'])
    min_samples_leaf = np.log10(grid_search.best_params_['min_samples_leaf'])
    max_leaf = np.log10(grid_search.best_params_['max_leaf_nodes'])
    param_grid = [{'min_samples_split': np.logspace(min_samples_split-0.3,
                                                    min_samples_split+0.7,
                                                    5, dtype=int),
                   'min_samples_leaf': np.logspace(min_samples_leaf-0.3,
                                                   min_samples_leaf+0.7,
                                                   5, dtype=int),
                   'max_leaf_nodes': np.logspace(max_leaf-0.3,
                                                 max_leaf+0.7,
                                                 5, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    # Third grid search to finetune hyperparameters
    min_samples_split = np.log10(grid_search.best_params_['min_samples_split'])
    min_samples_leaf = np.log10(grid_search.best_params_['min_samples_leaf'])
    max_leaf = np.log10(grid_search.best_params_['max_leaf_nodes'])
    param_grid = [{'min_samples_split': np.logspace(min_samples_split-0.1,
                                                    min_samples_split+0.1,
                                                    3, dtype=int),
                   'min_samples_leaf': np.logspace(min_samples_leaf-0.1,
                                                   min_samples_leaf+0.1,
                                                   3, dtype=int),
                   'max_leaf_nodes': np.logspace(max_leaf-0.1,
                                                 max_leaf+0.1,
                                                 3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def forest_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error'):
    """ Automated Selection of the hyparameters for Random Forest Regressor
    The selection is based on grid search.
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
    n_samples = X.shape[0]
    # First run with a random search
    param_grid = [{'n_estimators': [200, 400, 600, 800, 1000],
                   'max_features': np.linspace(0.2, 1.0, 5),
                   'bootstrap': [True, False]}]
    rand_search = RandomizedSearchCV(estimator, param_grid, cv=cv,
                                     scoring=scoring, random_state=42, n_jobs=-1)
    rand_search.fit(X, y)
    estimator = rand_search.best_estimator_
    # Second run with a grid search
    n_estimators = rand_search.best_params_['n_estimators']
    max_features = rand_search.best_params_['max_features']
    param_grid = [{'n_estimators': np.linspace(n_estimators-100, n_estimators+100,
                                               3, dtype=int),
                   'max_features': np.linspace(max_features-0.1, max_features+0.1, 5)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator = grid_search.best_estimator_
    # Third run with a grid search
    max_features = grid_search.best_params_['max_features']
    param_grid = [{'max_features': np.linspace(max_features-0.03, max_features+0.03, 7)}]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_
