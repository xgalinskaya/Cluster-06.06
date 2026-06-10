import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Kraljic Matrix Dashboard", layout="wide")

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
    df.columns = df.columns.str.strip()
    df['Order Value USD'] = df['Order Value USD'].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    return df

@st.cache_resource
def train_category_models(df):
    models = {}
    for category in df['Product_Category'].unique():
        cat_df = df[df['Product_Category'] == category].groupby('Supplier_ID').agg({'Order Value USD': 'sum', **{col: 'mean' for col in RISK_COLS}})
        scaler = MinMaxScaler()
        scaler.fit(cat_df[['Order Value USD']])
        models[category] = {
            'scaler': scaler, 
            'mean_spend': np.mean(scaler.transform(cat_df[['Order Value USD']])), 
            'mean_risk': np.mean(cat_df[RISK_COLS].mean(axis=1))
        }
    return models

def get_kraljic_quadrant(spend, risk, mean_spend, mean_risk):
    if spend > mean_spend and risk > mean_risk: return "Strategic"
    if spend > mean_spend and risk <= mean_risk: return "Leverage"
    if spend <= mean_spend and risk > mean_risk: return "Bottleneck"
    return "Non-Critical"

def main():
    if 'supp' not in st.session_state: st.session_state.supp = "All Suppliers"
    if 'time' not in st.session_state: st.session_state.time = "All Months"
    if 'weights' not in st.session_state: st.session_state['weights'] = [20, 20, 20, 20, 20]

    st.title("Sustainable Supply Chain: Category-Specific Kraljic Matrix")
    df = load_data()
    models = train_category_models(df)

    # --- SIDEBAR ---
    selected_cat = st.sidebar.selectbox("Select Product Category", sorted(df['Product_Category'].unique()))
    
    # --- RISK WEIGHTS ---
    st.sidebar.header("Risk Weights")
    preset = st.sidebar.selectbox(
        "Weight Profile",
        ["Custom", "Balanced", "Quality Focus", "Financial Focus", "Sustainability Focus", "Political Focus"]
    )

    if preset != "Custom":
        profiles = {
            "Balanced": [20, 20, 20, 20, 20],
            "Quality Focus": [50, 10, 10, 10, 20],
            "Financial Focus": [10, 50, 10, 10, 20],
            "Sustainability Focus": [10, 10, 50, 20, 10],
            "Political Focus": [10, 10, 10, 20, 50]
        }
        st.session_state['weights'] = profiles[preset]

    new_weights = []
    for i, col in enumerate(RISK_COLS):
        label = col.replace('_Score', '').replace('_', ' ')
        val = st.sidebar.slider(label, 0, 100, st.session_state['weights'][i], 5)
        new_weights.append(val)
    
    st.session_state['weights'] = new_weights
    total_weight = sum(new_weights)

    if total_weight != 100:
        st.sidebar.error(f"⚠️ Sum must be 100! Current: {total_weight}")
        st.stop()
    else:
        st.sidebar.success("✅ Weights normalized")

    weights_normalized = np.array(new_weights) / 100

    # --- PIPELINE ---
    subset = df[df['Product_Category'] == selected_cat]
    agg_df = subset.groupby('Supplier_ID').agg({'Order Value USD': 'sum', 'Country': 'first', **{col: 'mean' for col in RISK_COLS}}).reset_index()
    
    model = models[selected_cat]
    agg_df['Normalized_Spend'] = np.clip(model['scaler'].transform(agg_df[['Order Value USD']]), 0, 1)
    # Используем нормализованные веса
    agg_df['Weighted_Risk'] = (agg_df[RISK_COLS].values * weights_normalized).sum(axis=1)
    agg_df['Kraljic_Quadrant'] = agg_df.apply(lambda x: get_kraljic_quadrant(x['Normalized_Spend'], x['Weighted_Risk'], model['mean_spend'], model['mean_risk']), axis=1)

    # --- PLOT ---
    fig = px.scatter(
        agg_df, x="Normalized_Spend", y="Weighted_Risk", color="Kraljic_Quadrant",
        hover_data=['Supplier_ID'], range_x=[0, 1], range_y=[0, 1]
    )
    fig.add_vline(x=model['mean_spend'], line_dash="dash", line_color="gray")
    fig.add_hline(y=model['mean_risk'], line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()