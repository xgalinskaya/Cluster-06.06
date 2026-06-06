import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Настройка страницы
st.set_page_config(page_title="Kraljic Matrix Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Убедитесь, что сепаратор и десятичный разделитель соответствуют вашему CSV
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
    # Очистка колонки трат
    df['Order Value USD'] = (
        df['Order Value USD'].astype(str).str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False).astype(float)
    )
    return df

def main():
    st.title("Sustainable Supply Chain: Kraljic Matrix")
    df = load_data()

    # --- ПАНЕЛЬ ВВОДА ВЕСОВ ---
    st.sidebar.header("Веса рисков (в сумме 100)")
    w1 = st.sidebar.number_input("Performance Quality", 0, 100, 20)
    w2 = st.sidebar.number_input("Financial Risk", 0, 100, 20)
    w3 = st.sidebar.number_input("Sustainability Score", 0, 100, 20)
    w4 = st.sidebar.number_input("Standards Risk", 0, 100, 20)
    w5 = st.sidebar.number_input("Political Risk", 0, 100, 20)

    if (w1 + w2 + w3 + w4 + w5) != 100:
        st.error("Веса должны составлять 100%.")
        st.stop()

    # Список колонок рисков из вашего файла
risk_cols = [
        'Performance_Quality_Risk_Score', 
        'Financial_Risk_Score', 
        'Nachhaltigkeit_Risk_score', 
        'Standards Risks_Score', 
        'Risikoscore_Political_Risk_Score'
    ]

    # --- ОБУЧЕНИЕ МОДЕЛИ ---
        if 'kmeans_model' not in st.session_state:
        agg_dict = {'Order Value USD': 'sum'}
        for col in risk_cols:
            agg_dict[col] = 'mean'
            
        base_df = df.groupby('Supplier_ID').agg(agg_dict)
        
        scaler_s = MinMaxScaler()
        base_df['Normalized_Spend'] = scaler_s.fit_transform(base_df[['Order Value USD']])
        
        weights = np.array([w1, w2, w3, w4, w5]) / 100
        base_df['Weighted_Risk'] = (base_df[risk_cols].values * weights).sum(axis=1)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(base_df[['Normalized_Spend', 'Weighted_Risk']])
        
        st.session_state['kmeans_model'] = kmeans
        st.session_state['scaler'] = scaler_s
        st.session_state['weights'] = weights

    # --- АГРЕГАЦИЯ И ПРЕДСКАЗАНИЕ ---
    agg_dict = {'Order Value USD': 'sum'}
    for col in risk_cols:
        agg_dict[col] = 'mean'
    
    agg_df = df.groupby('Supplier_ID').agg(agg_dict).reset_index()

    agg_df['Normalized_Spend'] = st.session_state['scaler'].transform(agg_df[['Order Value USD']])
    agg_df['Weighted_Risk'] = (agg_df[risk_cols].values * st.session_state['weights']).sum(axis=1)
    
    agg_df['Cluster_ID'] = st.session_state['kmeans_model'].predict(agg_df[['Normalized_Spend', 'Weighted_Risk']])

    # Называем кластеры
    centroids = st.session_state['kmeans_model'].cluster_centers_
    cluster_map = {i: "" for i in range(4)}
    for i in range(4):
        s, r = centroids[i]
        if s > 0.5 and r > 0.5: cluster_map[i] = "Strategic"
        elif s > 0.5 and r <= 0.5: cluster_map[i] = "Leverage"
        elif s <= 0.5 and r > 0.5: cluster_map[i] = "Bottleneck"
        else: cluster_map[i] = "Non-Critical"
    
    agg_df['Kraljic_Quadrant'] = agg_df['Cluster_ID'].map(cluster_map)

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig = px.scatter(agg_df, x="Normalized_Spend", y="Weighted_Risk", color="Kraljic_Quadrant", hover_data=['Supplier_ID'])
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()