"""
Setup configuration for CardioMonoStack package.
"""

from setuptools import setup, find_packages

setup(
    name="cardiomonostack",
    version="1.0.0",
    author="CardioMonoStack Development Team",
    description=(
        "A modular ensemble learning framework for postoperative "
        "heart failure risk prediction after total hip arthroplasty."
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.0",
        "scikit-learn>=1.4",
        "lightgbm>=4.0",
        "xgboost>=2.0",
        "catboost>=1.2",
        "interpret>=0.5",
        "scipy>=1.11",
        "joblib>=1.3"
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
