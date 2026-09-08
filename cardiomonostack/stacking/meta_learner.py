"""
Stacking Meta Learner (L2)

Combines outputs from three heterogeneous
base learners.

Input:
    P_A
    P_B
    P_C

Output:
    P_L2
"""


import numpy as np


from sklearn.linear_model import LogisticRegression




class MetaLearner:
    """
    L2 stacked logistic regression.

    The learner receives only branch-level
    probability predictions.

    """



    def __init__(
        self,
        random_state=42
    ):


        self.random_state = random_state



        self.model = LogisticRegression(


            # L2 regularisation

            penalty="l2",

            C=1.0,


            # optimizer

            solver="lbfgs",


            # imbalance correction

            class_weight="balanced",


            max_iter=5000,


            random_state=random_state

        )



        self.is_fitted = False




    def fit(
        self,
        meta_features,
        y
    ):
        """
        Fit stacking meta learner.

        Parameters
        ----------
        meta_features :
            Matrix containing:

            [
                P_A,
                P_B,
                P_C
            ]

        y :
            Binary outcome

        """


        meta_features = np.asarray(
            meta_features
        )


        self.model.fit(

            meta_features,

            y

        )


        self.is_fitted = True


        return self




    def predict_proba(
        self,
        meta_features
    ):
        """
        Generate fused L2 probability.

        Returns
        -------
        ndarray

            P_L2

        """


        if not self.is_fitted:

            raise RuntimeError(
                "MetaLearner has not been fitted."
            )



        meta_features = np.asarray(
            meta_features
        )



        probability = (

            self.model
            .predict_proba(meta_features)[:,1]

        )


        return probability




    def predict(
        self,
        meta_features,
        threshold=0.437
    ):
        """
        Binary prediction.
        """


        probability = self.predict_proba(
            meta_features
        )


        return (
            probability >= threshold
        ).astype(int)



    def get_coefficients(
        self
    ):
        """
        Return stacking coefficients.

        Useful for model inspection.
        """


        if not self.is_fitted:

            raise RuntimeError(
                "MetaLearner has not been fitted."
            )


        return {

            "Branch_A": self.model.coef_[0][0],

            "Branch_B": self.model.coef_[0][1],

            "Branch_C": self.model.coef_[0][2],

            "Intercept": self.model.intercept_[0]

        }
