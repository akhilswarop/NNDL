import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Credit Risk Assessment")

# Sidebar to input parameters
loan_amount = st.sidebar.slider('Loan Amount', 5000, 50000, 15000)
income = st.sidebar.slider('Income', 20000, 200000, 50000)
debt_ratio = st.sidebar.slider('Debt-to-Income Ratio (%)', 0, 100, 20)

# Dummy model prediction
if st.button('Predict'):
    # You would use a trained model here
    risk_score = np.random.rand()
    if risk_score > 0.7:
        st.write("High Risk")
    else:
        st.write("Low Risk")
