import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Kraljic Matrix Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
    df['Order Value USD'] = (
        df['Order Value USD'].astype(str).str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False).astype(float)
    )
    return df

def main():
    st.title("Sustainable Supply Chain: Kraljic Matrix")
    df = load_data()

    risk_cols = [
        'Performance_Quality_Risk_Score', 
        'Financial_Risk_Score', 
        'Nachhaltigkeit_Risk_score', 
        'Standards Risks_Score', 
        'Political_Risk_Score'
    ]

    # --- 1. ОБУЧЕНИЕ НА ГОДОВЫХ ДАННЫХ (Фиксируем рамки) ---
    if 'kmeans' not in st.session_state:
        agg_full = df.groupby('Supplier_ID').agg({**{'Order Value USD': 'sum'}, **{c: 'mean' for c in risk_cols}})
        scaler = MinMaxScaler()
        agg_full['Normalized_Spend'] = scaler.fit_transform(agg_full[['Order Value USD']])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        agg_full['Weighted_Risk'] = (agg_full[risk_cols].values * weights).sum(axis=1)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(agg_full[['Normalized_Spend', 'Weighted_Risk']])
        st.session_state.update({'kmeans': kmeans, 'scaler': scaler, 'weights': weights})

    # --- 2. SIDEBAR ---
    st.sidebar.header("Risk Weights")
    w1, w2, w3, w4, w5 = [st.sidebar.number_input(l, 0, 100, 20) for l in ["Perf. Quality", "Financial", "Sustainability", "Standards", "Political"]]
    
    if (w1 + w2 + w3 + w4 + w5) == 100:
        st.session_state['weights'] = np.array([w1, w2, w3, w4, w5]) / 100
    
    selected_month = st.sidebar.selectbox("Select Month", sorted(df['Month'].unique()))
    
    # --- 3. ПРИМЕНЕНИЕ К МЕСЯЧНЫМ ДАННЫМ ---
    df_filtered = df[df['Month'] == selected_month].copy()
    agg_df = df_filtered.groupby('Supplier_ID').agg({**{'Order Value USD': 'sum'}, **{c: 'mean' for c in risk_cols}}).reset_index()
    
    # Защита от выхода за рамки 0-1 с помощью clip
    norm_spend = st.session_state['scaler'].transform(agg_df[['Order Value USD']])
    agg_df['Normalized_Spend'] = np.clip(norm_spend, 0, 1)
    
    agg_df['Weighted_Risk'] = (agg_df[risk_cols].values * st.session_state['weights']).sum(axis=1)
    agg_df['Cluster_ID'] = st.session_state['kmeans'].predict(agg_df[['Normalized_Spend', 'Weighted_Risk']])

    # Названия и цвета
    cluster_names = {0: "Non-Critical", 1: "Strategic", 2: "Leverage", 3: "Bottleneck"}
    agg_df['Kraljic_Quadrant'] = agg_df['Cluster_ID'].map(cluster_names)

    # --- 4. ВИЗУАЛИЗАЦИЯ ---
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            agg_df, x="Normalized_Spend", y="Weighted_Risk", color="Kraljic_Quadrant", 
            hover_data=['Supplier_ID'],
            category_orders={"Kraljic_Quadrant": ["Non-Critical", "Strategic", "Leverage", "Bottleneck"]}
        )
        event = st.plotly_chart(fig, on_select="rerun")

    with col2:
        st.subheader("Supplier Details")
        if event and event["selection"]["points"]:
            sel_id = event["selection"]["points"][0]["customdata"][0]
            data = agg_df[agg_df['Supplier_ID'] == sel_id].iloc[0]
            st.write(f"**Supplier:** {data['Supplier_ID']}")
            st.metric("Spend", f"{data['Order Value USD']:,.0f} $")
            for r in risk_cols:
                st.progress(data[r], text=r.replace('_Score', ''))
        else:
            st.info("Click on a point.")

if __name__ == "__main__":
    main()