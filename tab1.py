import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st


def get_importances(enc_df):
    X = enc_df.drop(['Rating'],axis=1)
    y = enc_df['Rating']
    feature_names = X.columns
    forest = RandomForestRegressor(random_state=0)
    forest.fit(X, y)

    importances = forest.feature_importances_

    forest_importances = pd.Series(importances, index=feature_names)

    return forest_importances

def get_corelations(enc_df):
    corr = enc_df.corr().round(2)
    return corr