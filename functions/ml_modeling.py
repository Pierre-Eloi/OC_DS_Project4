#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for data modeling
with machine learning algorithms."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
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
        gboost_reg = gboost_reg_best_params(X, y, GradientBoostingRegressor())
        xgboost_reg = xgboost_reg_best_params(X, y, XGBRegressor())
        mlp_reg = mlp_reg_best_params(X, y, MLPRegressor())
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
        xgboost_reg = XGBRegressor(random_state=42)
        mlp_reg = MLPRegressor(random_state=42)
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

def compare_models(X, y, estimators):
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
    for m in estimators:
        m_scores = cross_validate(m, X, y, scoring=scoring, n_jobs=-1)
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
    rand_search = RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter,
                                     scoring=scoring, random_state=42, n_jobs=-1)
    rand_search.fit(X, y)
    # Second search with a grid one
    threshold = rand_search.best_params_['selector__threshold']
    idx = np.where(thresholds==threshold)[0][0]
    if idx==0:
        param_grid = [{'selector__threshold': [thresholds[i] for i in range(idx, idx+4)]}]
    else:
        param_grid = [{'selector__threshold': [thresholds[i] for i in range(idx-4, idx+4)]}]
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, n_jobs=-1)
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
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        alpha = grid_search.best_params_['alpha']
        estimator = grid_search.best_estimator_
        # Second grid search
        param_grid = [{'alpha': np.linspace(alpha*0.5, alpha*5, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search
        alpha= grid_search.best_params_['alpha']
        param_grid = [{'alpha': np.linspace(alpha*5/6, alpha*7/6, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)

    else:
        # First grid search
        param_grid = [{'alpha': np.logspace(-1, 3, 5)}]
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        alpha = grid_search.best_params_['alpha']
        # Second grid search
        param_grid = [{'alpha': np.linspace(alpha*0.5, alpha*5, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search
        alpha= grid_search.best_params_['alpha']
        param_grid = [{'alpha': np.linspace(alpha*5/6, alpha*7/6, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
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
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Second search with a randomized search
        c = grid_search.best_params_['C']
        gamma = grid_search.best_params_['gamma']
        epsilon = grid_search.best_params_['epsilon']
        param_grid = [{'C': np.linspace(c, c*5, 10),
                       'gamma': np.linspace(gamma, gamma*5, 10),
                       'epsilon': np.linspace(epsilon, epsilon*5, 10)}]
        rand_search = RandomizedSearchCV(estimator, param_grid, n_iter=100,
                                         scoring=scoring, n_jobs=-1, random_state=42)
        rand_search.fit(X, y)
        # Third search with a grid search
        c = rand_search.best_params_['C']
        gamma = rand_search.best_params_['gamma']
        epsilon = rand_search.best_params_['epsilon']
        param_grid = [{'C': np.linspace(c*5/6, c*7/6, 3),
                       'gamma': np.linspace(gamma*5/6, gamma*7/6, 3),
                       'epsilon': np.linspace(epsilon*7/6, epsilon*7/6, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
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
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Second grid search
        c = grid_search.best_params_['C']
        epsilon = grid_search.best_params_['epsilon']
        param_grid = [{'C': np.linspace(c/2, c*5, 10),
                       'epsilon': np.linspace(epsilon/2, epsilon*5, 10)}]
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        # Third grid search
        c = grid_search.best_params_['C']
        epsilon = grid_search.best_params_['epsilon']
        param_grid = [{'C': np.linspace(c*5/6, c*7/6, 3),
                       'epsilon': np.linspace(epsilon*5/6, epsilon*7/6, 3)}]
        grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
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
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    best_n = grid_search.best_params_['n_neighbors']
    param_grid = [{'n_neighbors': np.linspace(best_n-1, best_n+1, 3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
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
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
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
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
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
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
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
    estimator = estimator.set_params(n_estimators=500,
                                     random_state=42,
                                     n_jobs=-1)
    # First search with a grid one
    param_grid = [{'max_features': np.linspace(0.2, 0.8, 4)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    # Second search with a grid one
    max_features = grid_search.best_params_['max_features']
    param_grid = [{'max_features': np.linspace(max_features-0.1, max_features+0.1, 5)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    # Second search with a grid one
    max_features = grid_search.best_params_['max_features']
    param_grid = [{'max_features': np.linspace(max_features-0.025, max_features+0.025, 3)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def gboost_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error'):
    """ Automated Selection of the hyparameters for Gradient Boosting Regressor
    Three differents steps are used for the selection.
    1) Using grid searches to fine-tune the major hyperparameters,
    the number of boosting stages (n_estimators) excepted.
    2) Fine-tune the n_estimators hyperparameter
    by using the early stopping technique.
    3) Fine-tune the subsample hyperparameter with a grid search
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        the Gradient boosting model to be improved
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    -----------
    Return:
        estimator
    """
    estimator = estimator.set_params(n_estimators=100,
                                     random_state=42)
    # 1) Fine-tune 'loss', 'learning_rate' and 'max_depth' hyperparmeters
    param_grid = [{'loss': ['ls', 'lad', 'huber'],
                   'learning_rate': [0.01, 0.1],
                   'max_depth': np.linspace(2, 10, 5, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator=grid_search.best_estimator_
    # Second grid search for fine-tuning
    learning_rate = grid_search.best_params_['learning_rate']
    max_depth = grid_search.best_params_['max_depth']
    param_grid = [{'learning_rate': np.linspace(learning_rate/2,
                                                learning_rate*5, 10),
                   'max_depth': np.linspace(max_depth-1,
                                            max_depth+1, 3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator=grid_search.best_estimator_
    # 2) Using early stopping to find the best 'n_estimators'
    min_val_error = float("inf")
    error_going_up = 0
    for n in range (50, 150):
        estimator.n_estimators = n
        val_error = -cross_val_score(estimator, X, y,
                                    scoring=scoring, n_jobs=-1).mean()
        if n == 50:
            estimator.warm_start = True
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up ==5:
                break # early stopping
    estimator.warm_start = False
    estimator.n_estimators = n - 5
    # 3) Fine-tune the 'subsample' hyperparameter
    param_grid = [{'subsample': np.linspace(0.6, 1, 5)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def xgboost_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error'):
    """ Automated Selection of the hyparameters
    for Extreme Gradient Boosting Regressor.
    Three differents steps are used for the selection.
    1) Using grid searches to fine-tune the major hyperparameters,
    the number of boosting stages (n_estimators) excepted.
    2) Fine-tune the n_estimators hyperparameter
    by using the early stopping technique.
    3) Fine-tune the subsample and colsample_bytree hyperparameters with a grid search
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        the Gradient boosting model to be improved
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    -----------
    Return:
        estimator
    """
    estimator = estimator.set_params(n_estimators=100,
                                     random_state=42)
    # 1) Fine-tune 'learning_rate' and 'max_depth' hyperparmeters
    param_grid = [{'learning_rate': [0.01, 0.1],
                   'max_depth': np.linspace(2, 10, 5, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator=grid_search.best_estimator_
    # Second grid search for fine-tuning
    learning_rate = grid_search.best_params_['learning_rate']
    max_depth = grid_search.best_params_['max_depth']
    param_grid = [{'learning_rate': np.linspace(learning_rate/2,
                                                learning_rate*5, 10),
                   'max_depth': np.linspace(max_depth+1,max_depth-1,
                                            3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator=grid_search.best_estimator_
    # 2) Using early stopping to find the best 'n_estimators'
    min_val_error = float("inf")
    error_going_up = 0
    for n in range (50, 150):
        estimator.n_estimators = n
        val_error = -cross_val_score(estimator, X, y,
                                    scoring=scoring, n_jobs=-1).mean()
        if n == 50:
            estimator.warm_start = True
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up ==5:
                break # early stopping
    estimator.warm_start = False
    estimator.n_estimators = n - 5
    # 3) Fine-tune the 'subsample' hyperparameter
    param_grid = [{'subsample': np.linspace(0.8, 1, 3),
                   'colsample_bytree': np.linspace(0.6, 1, 5)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def mlp_reg_best_params(X, y, estimator, scoring='neg_mean_squared_error'):
    """ Automated Selection of the hyparameters
    for Extreme Gradient Boosting Regressor.
    Three differents steps are used for the selection.
    1) Using grid searches to fine-tune the activation function and
    the solver hyperparameters
    2) Using grid searches to fine-tune the alpha,
    learning_rate, and the learning_rate_init hyperparameters.
    3) Learning_rate hyperparameter fine-tuning
    4) Get the optimum number of hidden layers
    -----------
    Parameters:
    X: Array
        the array object holding data
    y: Array
        the target
    estimator: estimator object
        the Gradient boosting model to be improved
    scoring: str or callable, default 'neg_mean_squared_error'
        all scorer objects follow the convention that
        higher return values are better than lower return values
    -----------
    Return:
        estimator
    """
    estimator = estimator.set_params(max_iter=500,
                                     early_stopping=True,
                                     random_state=42)
    # 1) Fine-tune 'activation' and 'solver' hyperparameters
    param_grid = [{'activation': ['logistic', 'tanh', 'relu'],
                   'solver': ['sgd', 'adam'],
                   'shuffle': [True, False]}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator=grid_search.best_estimator_
    # 2) Fine-tune the L2 penalty and the learning rate hyperparameters
    param_grid = [{'alpha': np.logspace(-5, -1, 5),
                   'learning_rate_init': np.logspace(-3, -1, 3),
                   'learning_rate': ['constant', 'invscaling', 'adaptive']}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    estimator=grid_search.best_estimator_
    # Second Grid Search
    alpha = grid_search.best_params_['alpha'] 
    learning_rate_init = grid_search.best_params_['learning_rate_init'] 
    param_grid = [{'alpha': np.linspace(alpha/2, alpha*5, 10),
                   'learning_rate_init': np.linspace(learning_rate_init/2,
                                                     learning_rate_init*5,
                                                     10)}]
    estimator=grid_search.best_estimator_
    # 3) Get the optimum number of hidden layers
    param_grid = [{'hidden_layer_sizes': np.logspace(0, 2, 3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    # Second Grid Search
    hidden_layer_sizes = grid_search.best_params_['hidden_layer_sizes'] 
    param_grid = [{'hidden_layer_sizes': np.linspace(hidden_layer_sizes/2,
                                                     hidden_layer_sizes*5,
                                                     10, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    # Third Grid Search
    hidden_layer_sizes = grid_search.best_params_['hidden_layer_sizes'] 
    param_grid = [{'hidden_layer_sizes': np.linspace(hidden_layer_sizes*5/6,
                                                     hidden_layer_sizes*7/6,
                                                     3, dtype=int)}]
    grid_search = GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_