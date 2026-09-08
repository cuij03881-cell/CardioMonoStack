"""
CardioMonoStack Example

Minimal example showing how to:

1. Prepare input data
2. Initialize transformer
3. Train CardioMonoStack
4. Generate risk probabilities


Note:
Real clinical datasets are not included
because source databases require controlled access.
"""


import pandas as pd
import numpy as np


from cardiomonostack import (
    CardioMonoStack,
    CardioMonoTransformer
)



# =====================================================
# 1. Load example data
# =====================================================


# Replace this with your own dataset

data = pd.DataFrame({


    "age": [
        72,
        80,
        68
    ],


    "sex": [
        1,
        0,
        1
    ],


    "Creatinine": [
        1.1,
        1.5,
        0.9
    ],


    "BUN": [
        22,
        30,
        15
    ],


    "Sodium": [
        137,
        135,
        140
    ],


    "Chloride": [
        102,
        100,
        104
    ],


    "Glucose": [
        130,
        160,
        110
    ],


    "HTN": [
        1,
        1,
        0
    ],


    "DM": [
        0,
        1,
        0
    ]

})



# Example labels

y = np.array([

    0,

    1,

    0

])



# =====================================================
# 2. Transform input
# =====================================================


transformer = CardioMonoTransformer()


X = transformer.fit_transform(
    data
)



# =====================================================
# 3. Initialize model
# =====================================================


model = CardioMonoStack()



# =====================================================
# 4. Train model
# =====================================================


model.fit(

    X,

    y

)



# =====================================================
# 5. Predict probability
# =====================================================


risk_probability = model.predict_proba(

    X

)



print(
    "Predicted heart failure risk:"
)


print(
    risk_probability
)



# =====================================================
# 6. Binary prediction
# =====================================================


prediction = model.predict(

    X

)



print(
    "Binary prediction:"
)


print(
    prediction
)
