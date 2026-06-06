import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Kraljic Matrix Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Загрузка данных
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
    df['Order Value USD'] = (
        df['Order Value USD'].astype(str).str.replace(' ', '', regex=False)
        .str.replace(',', '.', regex=False).astype(float)
    )
    return df

def main():
    st.title("Sustainable Supply Chain: Kraljic Matrix")
    df = load_data()

    # Риски для расчета
    risk_cols = [
        'Performance_Quality_Risk_Score', 
        'Financial_Risk_Score', 
        'Nachhaltigkeit_Risk_score', 
        'Standards Risks_Score', 
        'Political_Risk_Score'
    ]

    # --- 1. ОБУЧЕНИЕ НА ГОДОВЫХ ДАННЫХ (Фиксируем рамки кластеров) ---
    if 'kmeans' not in st.session_state:
        # Агрегация по всему году
        agg_full = df.groupby('Supplier_ID').agg({**{'Order Value USD': 'sum'}, **{c: 'mean' for c in risk_cols}})
        
        # Обучаем scaler на годовых данных
        scaler = MinMaxScaler()
        agg_full['Normalized_Spend'] = scaler.fit_transform(agg_full[['Order Value USD']])
        
        # Рассчитываем веса (фиксируем 20% как дефолт, чтобы модель обучилась)
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        agg_full['Weighted_Risk'] = (agg_full[risk_cols].values * weights).sum(axis=1)
        
        # Обучаем кластеризацию
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(agg_full[['Normalized_Spend', 'Weighted_Risk']])
        
        # Сохраняем модель и веса
        st.session_state.update({'kmeans': kmeans, 'scaler': scaler, 'weights': weights})

    # --- 2. SIDEBAR (Фильтры) ---
    st.sidebar.header("Risk Weights")
    w1, w2, w3, w4, w5 = [st.sidebar.number_input(label, 0, 100, 20) for label in ["Perf. Quality", "Financial Risk", "Sustainability", "Standards Risk", "Political Risk"]]
    
    if (w1 + w2 + w3 + w4 + w5) != 100:
        st.sidebar.error("Weights must sum to 100%.")
    else:
        st.session_state['weights'] = np.array([w1, w2, w3, w4, w5]) / 100

    selected_month = st.sidebar.selectbox("Select Month", sorted(df['Month'].unique()))
    
    # --- 3. ПРИМЕНЕНИЕ К МЕСЯЧНЫМ ДАННЫМ ---
    df_filtered = df[df['Month'] == selected_month].copy()
    agg_df = df_filtered.groupby('Supplier_ID').agg({**{'Order Value USD': 'sum'}, **{c: 'mean' for c in risk_cols}}).reset_index()
    
    # Используем ГОДОВОЙ scaler и ГОДОВЫЕ веса
    agg_df['Normalized_Spend'] = st.session_state['scaler'].transform(agg_df[['Order Value USD']])
    agg_df['Weighted_Risk'] = (agg_df[risk_cols].values * st.session_state['weights']).sum(axis=1)
    agg_df['Cluster_ID'] = st.session_state['kmeans'].predict(agg_df[['Normalized_Spend', 'Weighted_Risk']])

    # Названия квадрантов на основе ГОДОВЫХ центроидов
    centroids = st.session_state['kmeans'].cluster_centers_
    cluster_map = {i: ("Strategic" if s>0.5 and r>0.5 else "Leverage" if s>0.5 else "Bottleneck" if r>0.5 else "Non-Critical") for i in range(4) for s,r in [centroids[i]]}
    agg_df['Kraljic_Quadrant'] = agg_df['Cluster_ID'].map(cluster_map)

    # --- 4. ВИЗУАЛИЗАЦИЯ ---
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(agg_df, x="Normalized_Spend", y="Weighted_Risk", color="Kraljic_Quadrant", hover_data=['Supplier_ID'])
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
            st.info("Click on a point to see details.")

if __name__ == "__main__":
    main()