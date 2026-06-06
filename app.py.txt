import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 1. Page Configuration
st.set_page_config(page_title="Kraljic Matrix Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
    # ... [Keep your existing data cleaning logic for Order Value USD and Quarter] ...
    return df

def main():
    st.title("Sustainable Supply Chain: Kraljic Matrix")
    df = load_data()

    # --- INPUT FIELDS FOR WEIGHTS ---
    st.sidebar.header("Risk Component Weights (Total: 100)")
    col1, col2 = st.sidebar.columns(2)
    w1 = col1.number_input("Quality", 0, 100, 20)
    w2 = col2.number_input("Financial", 0, 100, 20)
    w3 = col1.number_input("Sustainability", 0, 100, 20)
    w4 = col2.number_input("Standards", 0, 100, 20)
    w5 = col1.number_input("Political", 0, 100, 20)

    if (w1 + w2 + w3 + w4 + w5) != 100:
        st.error("Weights must sum to 100.")
        st.stop()

    # --- DYNAMIC CLUSTERING BASED ON FULL YEAR ---
    # We calculate the fixed boundaries based on the full dataset once
    scaler_spend = MinMaxScaler(feature_range=(0, 1))
    scaler_risk = MinMaxScaler(feature_range=(0, 1))
    
    # Pre-calculate base centroids for the full year to keep logic consistent
    full_agg = df.groupby('Supplier_ID').agg({'Order Value USD': 'sum', 'Performance_Quality_Score': 'mean' ...}) # Add all score cols
    # ... Apply MinMaxScaling and KMeans here ...
    # Store these as session state variables to keep them static during filtering

    # --- FILTERING ---
    # Apply filters only to the display DataFrame, not the clustering logic
    
    # --- VISUALIZATION ---
    # Use config={"displayModeBar": True} and click events
    fig = px.scatter(df_f, x="Normalized_Spend", y="Weighted_Risk", ...)
    
    # To show the table on click:
    event = st.plotly_chart(fig, on_select="rerun")
    
    if event and event["selection"]["points"]:
        selected_idx = event["selection"]["points"][0]["point_index"]
        supplier_data = df_f.iloc[selected_idx]
        
        # Display the specific metrics as shown in your screenshot
        st.subheader("Punktedetails")
        # Use st.metric or st.dataframe to display specific fields
        st.write(f"Supplier: {supplier_data['Supplier_ID']}")
        # ... display remaining fields ...