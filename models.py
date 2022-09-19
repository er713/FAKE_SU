from random import choices
from tkinter.tix import Tree
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import ParameterGrid, GridSearchCV
import matplotlib.pyplot as plt
from lime import lime_tabular
import argparse
import re


feature_names = (
    ["#authors", "best_author_#art", "title_len", "text_len"]
    + [f"title_fake{i}" for i in range(10)]
    + [f"title_true{i}" for i in reversed(range(10))]
    + [f"text_fake{i}" for i in range(10)]
    + [f"text_true{i}" for i in reversed(range(10))]
)
label_names = ["true", "fake"]


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--svm", action="store_true")
    group.add_argument("-d", "--d_tree", action="store_true")
    group.add_argument("-l", "--logistic", action="store_true")
    # group.set_defaults(svm=True)
    parser.add_argument("-x", "--explain", action="store_true")
    return parser.parse_args()


def _run_grid_template(x, y, model, params):
    grid = GridSearchCV(
        model,
        params,
        scoring=make_scorer(accuracy_score),
        n_jobs=-1,
        cv=4,
        verbose=4,
    )
    grid = grid.fit(x, y)
    print("BEST PARAMS", grid.best_params_)
    return grid.best_estimator_


def prepare_svm_cv(x, y):
    """
    === old data:
        BEST PARAMS {'svm__C': 5.0, 'svm__class_weight': 'balanced', 'svm__coef0': 0.1, 'svm__degree': 4, 'svm__gamma': 'scale', 'svm__kernel': 'poly'}
        ACCURACY: 0.6941208640962537
    === second one:
        BEST PARAMS {'svm__C': 5.0, 'svm__class_weight': 'balanced', 'svm__coef0': 0.1, 'svm__degree': 4, 'svm__gamma': 'scale', 'svm__kernel': 'poly'}
    ACCURACY: 0.6945583811867652
    """
    model = Pipeline(steps=[("norm", Normalizer()), ("svm", SVC())])
    return _run_grid_template(
        x,
        y,
        model,
        [
            {
                "svm__kernel": ["linear"],
                "svm__C": [0.01, 0.5, 1.0, 5.0, 8.0],
                "svm__class_weight": [None, "balanced"],
            },
            {
                "svm__kernel": ["poly"],
                "svm__degree": [3, 4],
                "svm__coef0": [0.0, -0.1, 0.1],
                "svm__gamma": ["scale", "auto"],
                "svm__C": [0.5, 1.0, 5.0],
                "svm__class_weight": [None, "balanced"],
            },
            {
                "svm__kernel": ["rbf"],
                "svm__gamma": ["scale", "auto"],
                "svm__C": [0.5, 1.0, 5.0],
                "svm__class_weight": [None, "balanced"],
            },
        ],
    )


def get_svm():
    return make_pipeline(
        Normalizer(),
        SVC(
            C=5,
            class_weight="balanced",
            coef0=0.1,
            degree=4,
            kernel="poly",
            gamma="scale",
            probability=True,
        ),
    )


def prepare_tree_cv(x, y):
    """
    === old data:
        BEST PARAMS {'class_weight': None, 'criterion': 'log_loss', 'max_depth': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'n_estimators': 100}
        ACCURACY: 0.7859447634673229
    === second one:
        BEST PARAMS {'class_weight': None, 'criterion': 'log_loss', 'max_depth': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'n_estimators': 100}
        ACCURACY: 0.7875307629204266
    """
    model = RandomForestClassifier()
    return _run_grid_template(
        x,
        y,
        model,
        {
            "n_estimators": [100, 30, 60],
            "criterion": ["gini", "log_loss"],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 0.01],
            "min_impurity_decrease": [0.0, 0.01],
            "class_weight": [None, "balanced"],
        },
    )


def get_tree():
    return RandomForestClassifier(criterion="log_loss")


def prepare_logi_cv(x, y):
    """
    === old data:
        BEST PARAMS {'log__C': 0.01, 'log__class_weight': None, 'log__penalty': 'none', 'log__solver': 'lbfgs', 'log__warm_start': False}
        ACCURACY: 0.6971287940935192
    === second one:
        BEST PARAMS {'log__C': 0.01, 'log__class_weight': 'balanced', 'log__max_iter': 1000, 'log__penalty': 'none', 'log__solver': 'saga', 'log__warm_start': True}
        ACCURACY: 0.6934098988241728
    """
    model = Pipeline(steps=[("norm", Normalizer()), ("log", LogisticRegression())])
    return _run_grid_template(
        x,
        y,
        model,
        {
            "log__penalty": ["l2", "elasticnet", "none"],
            "log__C": [0.01, 0.5, 1.0, 5.0, 8.0],
            "log__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "log__class_weight": [None, "balanced"],
            "log__warm_start": [False, True],
            "log__max_iter": [
                1000,
            ],
        },
    )


def get_log_reg():
    return make_pipeline(
        Normalizer(),
        LogisticRegression(
            C=0.01,
            penalty="none",
            solver="saga",
            class_weight="balanced",
            warm_start=True,
        ),
    )


def _prepare_classification_lime(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))

    _lime = lime_tabular.LimeTabularExplainer(
        x_train, "classification", y_train, feature_names, class_names=label_names
    )
    model_name = re.findall(r"SVC|RandomForest|LogisticRegression", str(model))[0]
    for r in choices(list(range(x_train.shape[0])), k=10):
        explanation = _lime.explain_instance(
            x_train[r], model.predict_proba, num_features=20, labels=(int(y_train[r]),)
        )
        explanation.save_to_file(f"results/{model_name}_{r}.html")


if __name__ == "__main__":
    opts = parse_args()

    train = np.load("data/train2.npy")
    # print(np.any(np.isnan(train), axis=0))
    x_train, y_train = train[:, :-1], train[:, -1]

    if opts.explain:
        if opts.svm:
            print("==================== SVM ====================")
            model = get_svm()
        elif opts.d_tree:
            print("==================== TREE ====================")
            model = get_tree()
        elif opts.logistic:
            print("==================== LOGI ====================")
            model = get_log_reg()
        else:
            exit()

        test = np.load("data/test2.npy")
        x_test, y_test = test[:, :-1], test[:, -1]
        _prepare_classification_lime(x_train, y_train, x_test, y_test, model)

    else:
        if opts.svm:
            print("==================== SVM ====================")
            model = prepare_svm_cv(x_train, y_train)
        elif opts.d_tree:
            print("==================== TREE ====================")
            model = prepare_tree_cv(x_train, y_train)
        elif opts.logistic:
            print("==================== LOGI ====================")
            model = prepare_logi_cv(x_train, y_train)
        else:
            exit()

        test = np.load("data/test2.npy")
        x_test, y_test = test[:, :-1], test[:, -1]

        print("ACCURACY:", accuracy_score(y_test, model.predict(x_test)))
