"""
Monotonic Branch (Branch A)

LightGBM classifier with clinical monotonicity constraints.

This branch encodes prior clinical assumptions while
retaining nonlinear boosting capability.
"""


import numpy as np

from lightgbm import LGBMClassifier




class MonotonicBranch:
    """
    Branch A:
    Monotonic LightGBM learner.

    The constraint vector follows the final predictor order:

    [
        age,
        sex,
        Creatinine,
        BUN,
        Sodium,
        Chloride,
        Glucose,
        HTN,
        DM
    ]

    Constraint:

    age          +1
    Creatinine   +1
    BUN          +1
    Glucose      +1

    Others       0

    """



    def __init__(
        self,
        random_state=42
    ):

        self.random_state = random_state


        self.model = LGBMClassifier(

            # boosting parameters

            n_estimators=300,

            learning_rate=0.03,

            num_leaves=15,

            max_depth=4,


            # sampling

            subsample=0.8,

            colsample_bytree=0.8,


            # regularisation

            reg_lambda=2,


            # imbalance handling

            class_weight="balanced",


            random_state=random_state,


            objective="binary",

            verbosity=-1
        )


        # Clinical monotonic prior

        self.monotonic_constraints = [

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



        self.is_fitted = False




    def fit(
        self,
        X,
        y
    ):
        """
        Fit monotonic LightGBM.

        Parameters
        ----------
        X :
            Training predictors.

        y :
            Binary outcome.

        """



        self.model.fit(

            X,

            y,


            # monotonic constraints

            callbacks=None,


            categorical_feature=None,


            eval_metric="logloss",


            monotone_constraints=
                self.monotonic_constraints

        )



        self.is_fitted = True


        return self




    def predict_proba(
        self,
        X
    ):
        """
        Return probability output.

        Returns
        -------
        ndarray
            Risk probability of HF.
        """


        if not self.is_fitted:

            raise RuntimeError(
                "MonotonicBranch has not been fitted."
            )


        probability = (
            self.model
            .predict_proba(X)[:,1]
        )


        return probability




    def predict(
        self,
        X,
        threshold=0.437
    ):
        """
        Binary classification.
        """


        probability = self.predict_proba(
            X
        )


        return (
            probability >= threshold
        ).astype(int)
