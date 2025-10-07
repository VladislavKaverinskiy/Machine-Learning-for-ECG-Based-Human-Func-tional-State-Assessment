import os
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import hdbscan
from scipy.stats import f_oneway
from scipy.optimize import minimize
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score
import itertools
from sklearn.metrics import confusion_matrix
from itertools import product
from typing import List
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import train_test_split
import importlib
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*deprecation.*")


from sklearn.model_selection import train_test_split


def load_features_and_labels(csv_path='work_data/normalized..csv', class_col='class'):
    """
    Loads data from CSV, separates features and class labels.

    Parameters:
        csv_path (str): path to the CSV file.
        class_col (str): name of the column with class labels.

    Returns:
        X (pd.DataFrame): features.
        y (pd.Series): class labels.
    """
    df = pd.read_csv(csv_path)
    if class_col not in df.columns:
        raise ValueError(f"Column '{class_col}' is not found in file.")
    y = df[class_col]
    X = df.drop(columns=[class_col])
    names = list(df.columns.drop('class'))
    return X, y, names

def load_and_process(csv_path, exclude_classes=None, merge_classes=None, normalized=True):
    df = pd.read_csv(csv_path)

    # If there is a normalized and unnormalized option, you can select a column/frame.
    # Assume that all features except 'class' are features.
    # You can switch between normalized and non-normalized features in the feature name or another file.

    X = df.drop(columns=['class']).values
    y = df['class'].values

    # Eliminating classes
    mask = np.ones(len(y), dtype=bool)
    if exclude_classes:
        for cls in exclude_classes:
            mask &= (y != cls)
    X = X[mask]
    y = y[mask]

    # Combining classes
    if merge_classes:
        for new_cls, old_classes in merge_classes.items():
            for old_cls in old_classes:
                y[y == old_cls] = new_cls

    return X, y


def split_data(X, y, test_size=0.25, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def run_lazy_predict(X_train, X_test, y_train, y_test):
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)
    return models


def shap_analysis(model, X, feature_names=None, top_dependence=3):
    from sklearn.linear_model import Perceptron, LogisticRegression

    try:
        if isinstance(model, (Perceptron, LogisticRegression)):
            explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
    except Exception as e:
        print("Explainer error, try KernelExplainer:", e)
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
        shap_values = explainer.shap_values(shap.sample(X, 100))

    if isinstance(shap_values, list):
        sv_to_plot = shap_values[0]
    elif hasattr(shap_values, "values"):
        sv_to_plot = shap_values.values
    else:
        sv_to_plot = shap_values

    print("SHAP Summary Plot (beeswarm)")
    shap.summary_plot(sv_to_plot, features=X, feature_names=feature_names, plot_type="dot")

    print("SHAP Summary Bar Plot")
    shap.summary_plot(sv_to_plot, features=X, feature_names=feature_names, plot_type="bar")

    mean_abs_shap = np.abs(sv_to_plot).mean(axis=0)

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    top_indices = np.argsort(mean_abs_shap)[::-1][:top_dependence]

    for idx in top_indices:
        feat_name = feature_names[idx]
        print(f"SHAP Dependence Plot for feature '{feat_name}'")
        shap.dependence_plot(idx, sv_to_plot, X, feature_names=feature_names)

def run_lazy_predict_and_get_best(X_train, X_test, y_train, y_test):
    clf = LazyClassifier(verbose=True, ignore_warnings=True, custom_metric=None)
    models_df, _ = clf.fit(X_train, X_test, y_train, y_test)
    print(models_df)
    # models_df — DataFrame with model name index
    best_model_name = models_df['Accuracy'].idxmax()
    print(f"The best classifier by accuracy: {best_model_name}")
    return best_model_name

def import_model_by_name(model_name):
    mapping = {
        'RandomForestClassifier': ('sklearn.ensemble', 'RandomForestClassifier'),
        'GradientBoostingClassifier': ('sklearn.ensemble', 'GradientBoostingClassifier'),
        'SVC': ('sklearn.svm', 'SVC'),
        'KNeighborsClassifier': ('sklearn.neighbors', 'KNeighborsClassifier'),
        'LogisticRegression': ('sklearn.linear_model', 'LogisticRegression'),
        'LGBMClassifier': ('lightgbm', 'LGBMClassifier'),
        'XGBClassifier': ('xgboost', 'XGBClassifier'),
        # add others as needed
    }

    if model_name not in mapping:
        print(f"[!] Model '{model_name}' is not found. Available: {list(mapping.keys())}")
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression

    module_name, class_name = mapping[model_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def train_best_model(X_train, y_train, model_name):
    model_class = import_model_by_name(model_name)
    model = model_class()
    model.fit(X_train, y_train)
    return model


def preprocess_classes(X, y, exclude_classes=None, merge_classes=None):
    """
    Excludes and merges classes in a dataset.

    Parameters:
    - X: np.ndarray or pd.DataFrame - features
    - y: np.ndarray or pd.Series - class labels
    - exclude_classes: list[int] - list of classes to exclude
    - merge_classes: dict[int, list[int]] - a dictionary where the key is the new class and the value
      is a list of old classes to merge

    Returns:
    - X_new: np.ndarray - updated signs
    - y_new: np.ndarray - updated class labels
    """
    X = np.array(X)
    y = np.array(y)

    mask = np.ones(len(y), dtype=bool)

    # Eliminating classes
    if exclude_classes:
        for cls in exclude_classes:
            mask &= (y != cls)

    X_new = X[mask]
    y_new = y[mask]

    # Combining classes
    if merge_classes:
        for new_cls, old_classes in merge_classes.items():
            for old_cls in old_classes:
                y_new[y_new == old_cls] = new_cls

    return X_new, y_new


def shap_feature_importance_summary(model, X, feature_names=None, top_n=10):
    # Create an explainer (for LGBM, LinearExplainer or Explainer are better)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # shap_values.values.shape = (n_samples, n_features)
    # The mean absolute value across all samples is a measure of importance
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })

    df = df.sort_values('mean_abs_shap', ascending=False).head(top_n).reset_index(drop=True)

    # Gradation of significance by quantiles of 33% and 66%
    q_low = df['mean_abs_shap'].quantile(0.33)
    q_high = df['mean_abs_shap'].quantile(0.66)

    def importance_grade(val):
        if val >= q_high:
            return 'High'
        elif val >= q_low:
            return 'Middle'
        else:
            return 'Low'

    df['importance'] = df['mean_abs_shap'].apply(importance_grade)

    print(df[['feature', 'mean_abs_shap', 'importance']])
    return df


def load_clustered_data(folder_path):
    data_list = []
    label_list = []
    feature_names = None  # let's add a variable to store the names of the features

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv") or "noise" in filename.lower():
            continue  # skip outliers and non-csv

        digits = ''.join(filter(str.isdigit, filename))
        if digits == '':
            continue  # If there are no numbers in the file name, skip it.
        cluster_label = int(digits)

        df = pd.read_csv(os.path.join(folder_path, filename))

        if 'class' in df.columns:
            df = df.drop(columns=['class'])  # remove the original class label

        if feature_names is None:
            feature_names = list(df.columns)

        data_list.append(df)
        label_list.extend([cluster_label] * len(df))

    X = pd.concat(data_list, ignore_index=True)
    y = label_list
    return X, y, feature_names


if __name__ == "__main__":
    '''
    X, y, feature_names = load_features_and_labels()

    X, y = preprocess_classes(X, y,
                              exclude_classes=[5],
                              merge_classes={10: [1, 2, 4], 11: [3, 6]})
    '''

    X, y, feature_names = load_clustered_data("clusters_output/4 clusters")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    best_model_name = run_lazy_predict_and_get_best(X_train, X_test, y_train, y_test)

    # for 2 clusters
    # model = LGBMClassifier()
    # 4 clusters
    model = XGBClassifier()
    model.fit(X_train, y_train)

    shap_analysis(model, X_test, feature_names=feature_names, top_dependence=5)












