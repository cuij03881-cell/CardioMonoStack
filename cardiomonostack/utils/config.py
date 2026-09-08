"""
CardioMonoStack Configuration

Centralized model parameters.

All parameters correspond to the
final reported CardioMonoStack architecture.
"""


# =====================================================
# Feature configuration
# =====================================================


FEATURE_ORDER = [

    "age",

    "sex",

    "Creatinine",

    "BUN",

    "Sodium",

    "Chloride",

    "Glucose",

    "HTN",

    "DM"

]



# =====================================================
# Branch A
# Monotonic LightGBM
# =====================================================


LIGHTGBM_PARAMS = {


    "n_estimators": 300,

    "learning_rate": 0.03,

    "num_leaves": 15,

    "max_depth": 4,


    "subsample": 0.8,

    "colsample_bytree": 0.8,


    "reg_lambda": 2,


    "class_weight": "balanced",


    "objective": "binary",


    "random_state": 42,


    "verbosity": -1

}



# Clinical monotonic constraints

MONOTONIC_CONSTRAINTS = [

    1,   # age

    0,   # sex

    1,   # Creatinine

    1,   # BUN

    0,   # Sodium

    0,   # Chloride

    1,   # Glucose

    0,   # HTN

    0    # DM

]



# =====================================================
# Branch B
# CatBoost
# =====================================================


CATBOOST_PARAMS = {


    "iterations": 500,

    "depth": 5,

    "learning_rate": 0.03,


    "l2_leaf_reg": 5,


    "auto_class_weights": "Balanced",


    "boosting_type": "Ordered",


    "loss_function": "Logloss",


    "eval_metric": "Logloss",


    "random_seed": 42,


    "verbose": False

}




# =====================================================
# Branch C
# Explainable Boosting Machine
# =====================================================


EBM_PARAMS = {


    "interactions": 5,


    "learning_rate": 0.03,


    "max_rounds": 5000,


    "max_bins": 256,


    "validation_size": 0.15,


    "random_state": 42

}




# =====================================================
# L2 Meta Learner
# Logistic Regression
# =====================================================


META_PARAMS = {


    "penalty": "l2",


    "C": 1.0,


    "solver": "lbfgs",


    "class_weight": "balanced",


    "max_iter": 5000,


    "random_state": 42

}




# =====================================================
# L3 Residual Correction
# =====================================================


RESIDUAL_PARAMS = {


    "n_estimators": 150,


    "learning_rate": 0.03,


    "max_depth": 2,


    "subsample": 1.0,


    "loss": "squared_error",


    "random_state": 42

}




# =====================================================
# Deployment threshold
# =====================================================


DEFAULT_THRESHOLD = 0.437




# =====================================================
# Reproducibility
# =====================================================


RANDOM_STATE = 42
