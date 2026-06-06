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
    df = pd.read_csv('Merged dataset with Scores.csv', sep=';', decimal=',')
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

    # --- ОБУЧЕНИЕ МОДЕЛИ 1 РАЗ (ПО ВСЕМ ДАННЫМ) ---
    if 'kmeans_model' not in st.session_state:
        # Агрегируем данные по году (все имеющиеся данные)
        base_df = df.groupby('Supplier_ID').agg({
            'Order Value USD': 'sum',
            'Performance_Quality_Score': 'mean',
            'Financial_Risk_Score_Quarterly': 'mean',
            'Nachhaltigkeitsscore': 'mean',
            'Standards Risks_Score': 'mean',
            'Risikoscore Political': 'mean'
        })
        
        # Нормализация
        scaler_s = MinMaxScaler()
        base_df['Normalized_Spend'] = scaler_s.fit_transform(base_df[['Order Value USD']])
        
        # Расчет риска
        risk_cols = ['Performance_Quality_Score', 'Financial_Risk_Score_Quarterly', 'Nachhaltigkeitsscore', 'Standards Risks_Score', 'Risikoscore Political']
        weights = np.array([w1, w2, w3, w4, w5]) / 100
        base_df['Weighted_Risk'] = (base_df[risk_cols].values * weights).sum(axis=1)
        
        # Обучение K-means
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(base_df[['Normalized_Spend', 'Weighted_Risk']])
        
        # Сохраняем модель и центроиды в сессию
        st.session_state['kmeans_model'] = kmeans
        st.session_state['scaler'] = scaler_s
        st.session_state['weights'] = weights

    # --- ФИЛЬТРАЦИЯ И ПРИМЕНЕНИЕ КЛАСТЕРОВ ---
    # (Здесь вы можете добавить фильтры по месяцам, кластеры не изменятся)
    agg_df = df.groupby('Supplier_ID').agg({
        'Order Value USD': 'sum', 'Performance_Quality_Score': 'mean', 
        'Financial_Risk_Score_Quarterly': 'mean', 'Nachhaltigkeitsscore': 'mean',
        'Standards Risks_Score': 'mean', 'Risikoscore Political': 'mean'
    }).reset_index()

    agg_df['Normalized_Spend'] = st.session_state['scaler'].transform(agg_df[['Order Value USD']])
    agg_df['Weighted_Risk'] = (agg_df.iloc[:, 2:7].values * st.session_state['weights']).sum(axis=1)
    
    # Применяем ПРЕДСКАЗАНИЕ обученной модели
    agg_df['Cluster_ID'] = st.session_state['kmeans_model'].predict(agg_df[['Normalized_Spend', 'Weighted_Risk']])

    # Называем кластеры по координатам центроидов
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
    event = st.plotly_chart(fig, on_select="rerun")

    # --- ДЕТАЛИ ---
    if event and event["selection"]["points"]:
        idx = event["selection"]["points"][0]["point_index"]
        supplier = agg_df.iloc[idx]
        st.subheader("Punktedetails")
        st.write(f"Supplier: {supplier['Supplier_ID']} | Spend: {supplier['Order Value USD']:.0f} | Quadrant: {supplier['Kraljic_Quadrant']}")

if __name__ == "__main__":
    main()