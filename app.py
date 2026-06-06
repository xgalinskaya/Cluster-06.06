import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Kraljic Matrix Dashboard", layout="wide")

# Глобальные настройки рисков
RISK_COLS = [
    'Performance_Quality_Risk_Score', 
    'Financial_Risk_Score', 
    'Nachhaltigkeit_Risk_score', 
    'Standards Risks_Score', 
    'Political_Risk_Score'
]

@st.cache_data
def load_data():
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
    df['Order Value USD'] = (
        df['Order Value USD'].astype(str).str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False).astype(float)
    )
    return df

@st.cache_resource
def train_category_models(df):
    models_by_category = {}
    categories = df['Product_Category'].unique()
    
    for category in categories:
        cat_df = df[df['Product_Category'] == category].groupby('Supplier_ID').agg(
            {'Order Value USD': 'sum', **{col: 'mean' for col in RISK_COLS}}
        )
        
        scaler = MinMaxScaler()
        normalized_spend = scaler.fit_transform(cat_df[['Order Value USD']])
        
        features = np.column_stack([normalized_spend, cat_df[RISK_COLS].mean(axis=1)])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(features)
        
        models_by_category[category] = {'scaler': scaler, 'kmeans': kmeans}
    return models_by_category

def main():
    st.title("Sustainable Supply Chain: Category-Specific Kraljic Matrix")
    df = load_data()
    models = train_category_models(df)

    # --- SIDEBAR ---
    st.sidebar.header("Configuration")
    selected_cat = st.sidebar.selectbox("Select Product Category", sorted(df['Product_Category'].unique()))
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", ["All Months"] + sorted(df['Month'].unique().astype(str).tolist()))
    
    # Сброс и поиск
    if st.sidebar.button("Reset Selection"):
        st.session_state['search_id'] = ""
    
    search_id = st.sidebar.text_input("Search Supplier ID:", key='search_id')

    st.sidebar.header("Risk Weights")
    w_vals = [st.sidebar.number_input(col.replace('_', ' '), 0, 100, 20) for col in RISK_COLS]
    weights = np.array(w_vals) / 100

    # --- DATA PIPELINE ---
    subset = df[df['Product_Category'] == selected_cat].copy()
    if selected_timeframe != "All Months":
        subset = subset[subset['Month'].astype(str) == selected_timeframe]
    
    agg_df = subset.groupby('Supplier_ID').agg(
        {'Order Value USD': 'sum', **{col: 'mean' for col in RISK_COLS}}
    ).reset_index()
    
    model_bundle = models[selected_cat]
    norm_spend = np.clip(model_bundle['scaler'].transform(agg_df[['Order Value USD']]), 0, 1)
    weighted_risk = (agg_df[RISK_COLS].values * weights).sum(axis=1)
    
    agg_df['Normalized_Spend'] = norm_spend
    agg_df['Weighted_Risk'] = weighted_risk
    agg_df['Cluster_ID'] = model_bundle['kmeans'].predict(np.column_stack([norm_spend, weighted_risk]))
    agg_df['Kraljic_Quadrant'] = agg_df['Cluster_ID'].map({0: "Non-Critical", 1: "Strategic", 2: "Leverage", 3: "Bottleneck"})

    # --- VISUALIZATION ---
    fig = px.scatter(
        agg_df, x="Normalized_Spend", y="Weighted_Risk", color="Kraljic_Quadrant",
        hover_data=['Supplier_ID'], range_x=[0, 1], range_y=[0, 1],
        category_orders={"Kraljic_Quadrant": ["Non-Critical", "Strategic", "Leverage", "Bottleneck"]}
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='White')))
    fig.update_layout(height=600, width=900)
    
    event = st.plotly_chart(fig, on_select="rerun")
    
    # Определение выбранного поставщика (через клик или поиск)
    sel_id = None
    if event and event["selection"]["points"]:
        sel_id = event["selection"]["points"][0]["customdata"][0]
    elif search_id and search_id in agg_df['Supplier_ID'].values:
        sel_id = search_id

    # --- DETAILS ---
    if sel_id:
        data = agg_df[agg_df['Supplier_ID'] == sel_id].iloc[0]
        st.write(f"### Supplier: {data['Supplier_ID']}")
        st.metric("Spend", f"{data['Order Value USD']:,.0f} $")
        
        for r in RISK_COLS:
            score = data[r]
            label = r.replace('_Score', '').replace('_', ' ')
            st.write(f"**{label}**: {score:.2f}")
            st.progress(score)
    else:
        st.info("Click a point or enter a Supplier ID to see details.")

if __name__ == "__main__":
    main()