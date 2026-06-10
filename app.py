import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Kraljic Matrix Dashboard", layout="wide")

# Ensure these keys exactly match the column headers in your CSV file
RISK_COLS = [
    'Performance_Quality_Risk_Score',
    'Financial_Risk_Score', 
    'Sustainability_Risk_score', 
    'Standards Risk_Score', 
    'Political_Risk_Score'
]

@st.cache_data
def load_data():
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
    # Strip whitespace from column names to prevent KeyErrors
    df.columns = df.columns.str.strip()
    df['Order Value USD'] = (
        df['Order Value USD'].astype(str).str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False).astype(float)
    )
    return df

@st.cache_resource
def train_category_models(df):
    models = {}
    for category in df['Product_Category'].unique():
        # Aggregate data by Supplier_ID for each category
        cat_df = df[df['Product_Category'] == category].groupby('Supplier_ID').agg(
            {'Order Value USD': 'sum', **{col: 'mean' for col in RISK_COLS}}
        )
        scaler = MinMaxScaler()
        normalized_spend = scaler.fit_transform(cat_df[['Order Value USD']])
        
        mean_spend = np.mean(normalized_spend)
        mean_risk = np.mean(cat_df[RISK_COLS].mean(axis=1))
        
        models[category] = {'scaler': scaler, 'mean_spend': mean_spend, 'mean_risk': mean_risk}
    return models

def get_kraljic_quadrant(spend, risk, mean_spend, mean_risk):
    if spend > mean_spend and risk > mean_risk: return "Strategic"
    if spend > mean_spend and risk <= mean_risk: return "Leverage"
    if spend <= mean_spend and risk > mean_risk: return "Bottleneck"
    return "Non-Critical"

def main():
    if 'selected_point' not in st.session_state: st.session_state['selected_point'] = None
    if 'weights' not in st.session_state: st.session_state['weights'] = [20, 20, 20, 20, 20]

    st.title("Sustainable Supply Chain: Category-Specific Kraljic Matrix")
    df = load_data()
    
    # Check if all required risk columns exist in the dataframe
    for col in RISK_COLS:
        if col not in df.columns:
            st.error(f"Column '{col}' not found. Available columns: {df.columns.tolist()}")
            st.stop()

    models = train_category_models(df)

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("Configuration")
    selected_cat = st.sidebar.selectbox("Select Product Category", sorted(df['Product_Category'].unique()))
    
    supplier_list = ["All Suppliers"] + sorted(df["Supplier_ID"].unique().tolist())
    selected_supplier = st.sidebar.selectbox("Select Supplier", supplier_list)
    
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", ["All Months"] + sorted(df['Month'].unique().astype(str).tolist()))
    
    if st.sidebar.button("Reset Selection"):
        st.session_state['selected_point'] = None
        st.rerun()

    # --- RISK WEIGHTS ---
    st.sidebar.header("Risk Weights")
    # ... (weight profiles logic remains the same) ...

    # --- DATA PIPELINE ---
    subset = df[df['Product_Category'] == selected_cat].copy()
    if selected_timeframe != "All Months":
        subset = subset[subset['Month'].astype(str) == selected_timeframe]
    
    agg_df = subset.groupby('Supplier_ID').agg(
        {'Order Value USD': 'sum', 'Country': 'first', **{col: 'mean' for col in RISK_COLS}}
    ).reset_index()
    
    model = models[selected_cat]
    norm_spend = np.clip(model['scaler'].transform(agg_df[['Order Value USD']]), 0, 1)
    weights = np.array(st.session_state['weights']) / 100
    agg_df['Weighted_Risk'] = (agg_df[RISK_COLS].values * weights).sum(axis=1)
    
    # --- DETAILS SECTION ---
    sel_id = selected_supplier if selected_supplier != "All Suppliers" else st.session_state.get('selected_point')
    
    if sel_id:
        supplier_data = agg_df[agg_df['Supplier_ID'] == sel_id]
        if not supplier_data.empty:
            data = supplier_data.iloc[0]
            st.write(f"### Supplier: {data['Supplier_ID']}")
            st.metric("Spend", f"{data['Order Value USD']:,.0f} $")
            for r in RISK_COLS:
                st.write(f"**{r.replace('_Score', '').replace('_', ' ')}**: {data[r]:.2f}")
                st.progress(data[r])
        else:
            # Display warning if no orders found for the specific selection
            st.warning(f"No orders found for supplier {sel_id} in the selected period and category.")
    else:
        st.info("Click a point on the chart or select a Supplier ID to see details.")

if __name__ == "__main__":
    main()