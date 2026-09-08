"""
Residual Logit Correction Layer (L3)

Feature-dependent probability refinement.

Input:
    X
    P_L2

Output:
    P_final
"""


import numpy as np


from sklearn.ensemble import GradientBoostingRegressor




class ResidualLogitCorrection:
    """
    L3 residual correction model.

    The correction is performed in logit space.

    """



    def __init__(
        self,
        random_state=42
    ):


        self.random_state = random_state



        self.model = GradientBoostingRegressor(


            # boosting parameters

            n_estimators=150,


            learning_rate=0.03,


            max_depth=2,


            subsample=1.0,


            random_state=random_state,


            loss="squared_error"

        )


        self.is_fitted = False




    @staticmethod
    def _logit(
        probability
    ):
        """
        Convert probability to log-odds.
        """


        eps = 1e-7


        probability = np.clip(
            probability,
            eps,
            1-eps
        )


        return np.log(
            probability /
            (1-probability)
        )




    @staticmethod
    def _sigmoid(
        value
    ):
        """
        Convert logit to probability.
        """


        return (
            1 /
            (1 + np.exp(-value))
        )




    def fit(
        self,
        X,
        p_l2,
        y
    ):
        """
        Fit residual correction model.

        Parameters
        ----------
        X :
            Original predictor matrix.

        p_l2 :
            L2 stacking probability.

        y :
            Binary outcome.

        """


        p_l2 = np.asarray(
            p_l2
        )


        y = np.asarray(
            y
        )


        # Residual target

        residual_target = (
            y - p_l2
        )



        self.model.fit(

            X,

            residual_target

        )


        self.is_fitted = True


        return self




    def predict_proba(
        self,
        X,
        p_l2
    ):
        """
        Generate final corrected probability.

        Parameters
        ----------
        X :
            Predictor matrix.

        p_l2 :
            L2 probability.

        Returns
        -------
        ndarray

            P_final

        """


        if not self.is_fitted:

            raise RuntimeError(
                "ResidualLogitCorrection has not been fitted."
            )



        # L2 probability to logit

        z_l2 = self._logit(
            p_l2
        )



        # patient-specific correction

        correction = self.model.predict(
            X
        )



        # corrected logit

        z_final = (
            z_l2 +
            correction
        )



        # back transformation

        p_final = self._sigmoid(
            z_final
        )


        return p_final




    def predict(
        self,
        X,
        p_l2,
        threshold=0.437
    ):
        """
        Binary prediction.
        """


        probability = self.predict_proba(
            X,
            p_l2
        )


        return (
            probability >= threshold
        ).astype(int)



    def get_residual_model(
        self
    ):
        """
        Return fitted residual model.

        Useful for model inspection.
        """


        if not self.is_fitted:

            raise RuntimeError(
                "Residual correction has not been fitted."
            )


        return self.model
