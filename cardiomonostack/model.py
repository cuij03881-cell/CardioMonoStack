"""
CardioMonoStack
A modular ensemble framework for postoperative heart failure risk prediction.

Architecture:
L1:
    Branch A - Monotonic LightGBM
    Branch B - CatBoost boosting
    Branch C - Explainable Boosting Machine (EBM)

L2:
    Logistic stacking meta-learner

L3:
    Feature-dependent residual logit correction
"""


import numpy as np


from .branches.monotonic_branch import MonotonicBranch
from .branches.boosting_branch import BoostingBranch
from .branches.ebm_branch import EBMBranch

from .stacking.meta_learner import MetaLearner
from .correction.residual_logit import ResidualLogitCorrection



class CardioMonoStack:
    """
    Main CardioMonoStack estimator.

    Parameters
    ----------
    random_state : int
        Random seed.

    Attributes
    ----------
    branch_a :
        Monotonic boosting branch

    branch_b :
        Unconstrained boosting branch

    branch_c :
        Explainable boosting branch

    meta :
        L2 stacking learner

    residual :
        L3 residual logit correction model
    """



    def __init__(
        self,
        random_state=42
    ):

        self.random_state = random_state


        # L1 branches

        self.branch_a = MonotonicBranch(
            random_state=random_state
        )

        self.branch_b = BoostingBranch(
            random_state=random_state
        )

        self.branch_c = EBMBranch(
            random_state=random_state
        )


        # L2 stacking

        self.meta = MetaLearner(
            random_state=random_state
        )


        # L3 residual correction

        self.residual = ResidualLogitCorrection(
            random_state=random_state
        )



        self.is_fitted = False



    def fit(
        self,
        X,
        y
    ):
        """
        Fit CardioMonoStack model.

        Parameters
        ----------
        X : array-like
            Predictor matrix.

        y : array-like
            Binary outcome.

        Returns
        -------
        self
        """


        # ==============================
        # L1: Train three branches
        # ==============================


        self.branch_a.fit(
            X,
            y
        )


        self.branch_b.fit(
            X,
            y
        )


        self.branch_c.fit(
            X,
            y
        )



        # Obtain branch-level probabilities

        p_a = self.branch_a.predict_proba(
            X
        )

        p_b = self.branch_b.predict_proba(
            X
        )

        p_c = self.branch_c.predict_proba(
            X
        )



        # ==============================
        # L2: Stacking layer
        # ==============================


        meta_features = np.column_stack(
            [
                p_a,
                p_b,
                p_c
            ]
        )


        self.meta.fit(
            meta_features,
            y
        )


        p_l2 = self.meta.predict_proba(
            meta_features
        )



        # ==============================
        # L3: Residual logit correction
        # ==============================


        self.residual.fit(
            X,
            p_l2,
            y
        )



        self.is_fitted = True


        return self



    def predict_proba(
        self,
        X
    ):
        """
        Generate final risk probability.

        Returns
        -------
        numpy.ndarray
            Probability of postoperative heart failure.
        """


        if not self.is_fitted:
            raise RuntimeError(
                "CardioMonoStack has not been fitted."
            )



        # L1 predictions

        p_a = self.branch_a.predict_proba(
            X
        )

        p_b = self.branch_b.predict_proba(
            X
        )

        p_c = self.branch_c.predict_proba(
            X
        )



        # L2 fusion

        meta_features = np.column_stack(
            [
                p_a,
                p_b,
                p_c
            ]
        )


        p_l2 = self.meta.predict_proba(
            meta_features
        )



        # L3 correction

        p_final = self.residual.predict_proba(
            X,
            p_l2
        )


        return p_final



    def predict(
        self,
        X,
        threshold=0.437
    ):
        """
        Binary prediction using predefined threshold.

        Default threshold:
        INSPIRE Youden-index derived threshold.
        """


        probability = self.predict_proba(
            X
        )


        return (
            probability >= threshold
        ).astype(int)
