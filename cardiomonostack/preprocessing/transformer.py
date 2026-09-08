"""
CardioMonoStack Input Transformer

Standardizes input format before model prediction.

Expected predictors:

age
sex
Creatinine
BUN
Sodium
Chloride
Glucose
HTN
DM

"""


import pandas as pd
import numpy as np




class CardioMonoTransformer:
    """
    Input formatter for CardioMonoStack.


    Responsibilities:

    1. Check required variables
    2. Preserve predictor order
    3. Convert categorical variables
    4. Return model-ready matrix

    """



    REQUIRED_FEATURES = [

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




    def __init__(self):

        self.feature_order = (
            self.REQUIRED_FEATURES.copy()
        )



        self.is_fitted = False




    def fit(
        self,
        X
    ):
        """
        Fit transformer.

        For deployment consistency,
        feature order is locked here.

        """


        self._check_features(
            X
        )


        self.is_fitted = True


        return self




    def transform(
        self,
        X
    ):
        """
        Transform input dataframe.

        Returns
        -------
        pandas.DataFrame

        """



        if not self.is_fitted:

            raise RuntimeError(
                "Transformer has not been fitted."
            )



        self._check_features(
            X
        )



        X = X.copy()



        # --------------------------
        # Binary coding
        # --------------------------


        X["sex"] = (
            X["sex"]
            .astype(int)
        )


        X["HTN"] = (
            X["HTN"]
            .astype(int)
        )


        X["DM"] = (
            X["DM"]
            .astype(int)
        )



        # --------------------------
        # Keep locked order
        # --------------------------


        X = X[
            self.feature_order
        ]



        return X




    def fit_transform(
        self,
        X
    ):

        self.fit(
            X
        )


        return self.transform(
            X
        )




    def _check_features(
        self,
        X
    ):
        """
        Check whether all required
        predictors are present.

        """


        if not isinstance(
            X,
            pd.DataFrame
        ):

            raise TypeError(
                "Input must be pandas DataFrame."
            )



        missing = [

            col

            for col in self.REQUIRED_FEATURES

            if col not in X.columns

        ]



        if len(missing) > 0:

            raise ValueError(

                f"Missing required features: {missing}"

            )
