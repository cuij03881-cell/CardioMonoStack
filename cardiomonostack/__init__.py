"""
CardioMonoStack

A modular ensemble learning framework for
postoperative heart failure risk prediction
after total hip arthroplasty.

Architecture:

L1:
    - Monotonic LightGBM branch
    - CatBoost boosting branch
    - Explainable Boosting Machine branch

L2:
    - Logistic stacking meta learner

L3:
    - Residual logit correction

"""


from .model import CardioMonoStack

from .preprocessing.transformer import (
    CardioMonoTransformer
)



__version__ = "1.0.0"



__author__ = (
    "CardioMonoStack Development Team"
)



__all__ = [

    "CardioMonoStack",

    "CardioMonoTransformer"

]
