"""
EBM Branch (Branch C)

Explainable Boosting Machine learner.

This branch provides an interpretable nonlinear
representation using additive shape functions
and selected pairwise interactions.
"""


import numpy as np


from interpret.glassbox import (
    ExplainableBoostingClassifier
)




class EBMBranch:
    """
    Branch C:
    Explainable Boosting Machine.

    Learns:
        - univariate feature effects
        - selected pairwise interactions

    Output:
        P_C

    """



    def __init__(
        self,
        random_state=42
    ):


        self.random_state = random_state



        self.model = ExplainableBoostingClassifier(


            # pairwise interaction terms

            interactions=5,


            # optimization

            learning_rate=0.03,


            max_rounds=5000,


            # discretization

            max_bins=256,


            # validation split

            validation_size=0.15,


            random_state=random_state

        )



        self.is_fitted = False




    def fit(
        self,
        X,
        y
    ):
        """
        Fit EBM model.

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
        Return EBM probability.

        Output:
            P_C

        """


        if not self.is_fitted:

            raise RuntimeError(
                "EBMBranch has not been fitted."
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



    def explain_global(
        self
    ):
        """
        Return global explanation.

        Used for:
        - feature effect visualization
        - interpretation analysis

        """


        if not self.is_fitted:

            raise RuntimeError(
                "EBMBranch has not been fitted."
            )


        return self.model.explain_global()



    def explain_local(
        self,
        X
    ):
        """
        Return patient-level explanation.
        """


        if not self.is_fitted:

            raise RuntimeError(
                "EBMBranch has not been fitted."
            )


        return self.model.explain_local(
            X
        )
