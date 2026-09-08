"""
Boosting Branch (Branch B)

Unconstrained CatBoost learner.

This branch captures nonlinear relationships without
monotonic restrictions.
"""


import numpy as np

from catboost import CatBoostClassifier




class BoostingBranch:
    """
    Branch B:
    Unconstrained CatBoost boosting model.

    Unlike Branch A, this branch does not impose
    clinical monotonic constraints.

    """



    def __init__(
        self,
        random_state=42
    ):

        self.random_state = random_state


        self.model = CatBoostClassifier(


            # boosting configuration

            iterations=500,

            depth=5,

            learning_rate=0.03,


            # regularisation

            l2_leaf_reg=5,


            # class imbalance

            auto_class_weights="Balanced",


            # reproducibility

            random_seed=random_state,


            # binary classification

            loss_function="Logloss",


            eval_metric="Logloss",


            # ordered boosting

            boosting_type="Ordered",


            verbose=False

        )


        self.is_fitted = False




    def fit(
        self,
        X,
        y
    ):
        """
        Fit CatBoost branch.

        Parameters
        ----------
        X :
            Predictor matrix.

        y :
            Binary outcome.

        """


        self.model.fit(

            X,

            y

        )


        self.is_fitted = True


        return self




    def predict_proba(
        self,
        X
    ):
        """
        Return branch probability.

        Output:
        P_B

        """


        if not self.is_fitted:

            raise RuntimeError(
                "BoostingBranch has not been fitted."
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
        Binary prediction.
        """


        probability = self.predict_proba(
            X
        )


        return (
            probability >= threshold
        ).astype(int)
