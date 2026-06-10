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
    # 1. Инициализация (всегда в начале)
    if 'supp' not in st.session_state: st.session_state.supp = "All Suppliers"
    if 'time' not in st.session_state: st.session_state.time = "All Months"
    if 'weights' not in st.session_state: st.session_state['weights'] = [20, 20, 20, 20, 20]
        
    st.title("Sustainable Supply Chain: Category-Specific Kraljic Matrix")
    df = load_data()
    models = train_category_models(df)

    # Sidebar
    selected_cat = st.sidebar.selectbox("Select Product Category", sorted(df['Product_Category'].unique()))
    if st.sidebar.button("Reset Selection"):
        st.session_state.supp = "All Suppliers"
        st.session_state.time = "All Months"
        st.rerun()

    supp_options = ["All Suppliers"] + sorted(df["Supplier_ID"].unique().tolist())
    time_options = ["All Months"] + sorted(df['Month'].astype(str).unique().tolist())

    selected_supplier = st.sidebar.selectbox("Select Supplier", supp_options, key="supp")
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", time_options, key="time")

    # --- RISK WEIGHTS ---
st.sidebar.header("Risk Weights")
preset = st.sidebar.selectbox(
    "Weight Profile",
    ["Custom", "Balanced", "Quality Focus", "Financial Focus", "Sustainability Focus", "Political Focus"]
)

# 2. FIXED: Consistent indentation (4 spaces)
    if preset != "Custom":
    profiles = {
        "Balanced": [20, 20, 20, 20, 20],
        "Quality Focus": [50, 10, 10, 10, 20],
        "Financial Focus": [10, 50, 10, 10, 20],
        "Sustainability Focus": [10, 10, 50, 20, 10],
        "Political Focus": [10, 10, 10, 20, 50]
    }
    st.session_state['weights'] = profiles[preset]

# 3. Sliders
new_weights = []
for i, col in enumerate(RISK_COLS):
    label = col.replace('_Score', '').replace('_', ' ')
    # Using the initialized session_state value safely
    val = st.sidebar.slider(label, 0, 100, int(st.session_state['weights'][i]), 5, key=f"slider_{i}")
    new_weights.append(val)

# 4. Calculation
total_weight = sum(new_weights)

# Save updates
    if new_weights != st.session_state['weights']:
    st.session_state['weights'] = new_weights
    st.rerun()

# 5. Validation (Aligned correctly)
    if total_weight != 100:
    st.sidebar.warning(f"⚠️ The sum of weights must be 100. Current: {total_weight}")
    weights_normalized = np.array(new_weights) / (total_weight if total_weight != 0 else 1)
    else:
    st.sidebar.success("✅ Weights are balanced")
    weights_normalized = np.array(new_weights) / 100

    # Pipeline
    subset = df[(df['Product_Category'] == selected_cat)]
    if selected_timeframe != "All Months": subset = subset[subset['Month'].astype(str) == selected_timeframe]

    if subset.empty:
        st.warning("No orders found for the selected category or period.")
        return

# 1. Агрегация данных
    agg_df = subset.groupby('Supplier_ID').agg({
        'Order Value USD': 'sum', 
        'Country': 'first', 
        **{col: 'mean' for col in RISK_COLS}
    }).reset_index()

    # 2. Используем ЗАФИКСИРОВАННУЮ модель для конкретной категории
    model = models[selected_cat]

    # 3. НОРМАЛИЗАЦИЯ: используем именно тот scaler, который был обучен на годовых данных
    # Это гарантирует, что "0" и "1" по оси X всегда будут соответствовать годовым максимумам
    agg_df['Normalized_Spend'] = np.clip(model['scaler'].transform(agg_df[['Order Value USD']]), 0, 1)

    # 4. РИСК: используем нормализованные веса (weights_normalized)
    agg_df['Weighted_Risk'] = (agg_df[RISK_COLS].values * weights_normalized).sum(axis=1)

    # 5. КВАДРАНТЫ: используем ФИКСИРОВАННЫЕ пороги mean_spend и mean_risk
    # Это фиксирует сетку, она не будет "плавать" при фильтрации
    agg_df['Kraljic_Quadrant'] = agg_df.apply(
        lambda x: get_kraljic_quadrant(
            x['Normalized_Spend'], 
            x['Weighted_Risk'], 
            model['mean_spend'], 
            model['mean_risk']
        ), axis=1
    )

    # Plot
    plot_df = agg_df[agg_df["Supplier_ID"] == selected_supplier] if selected_supplier != "All Suppliers" else agg_df
    fig = px.scatter(
        plot_df, x="Normalized_Spend", y="Weighted_Risk", color="Kraljic_Quadrant",
        hover_data=['Supplier_ID'], range_x=[0, 1], range_y=[0, 1],
        category_orders={"Kraljic_Quadrant": ["Strategic", "Leverage", "Bottleneck", "Non-Critical"]}
    )
    fig.update_traces(marker=dict(size=16, line=dict(width=2, color='White')))
    fig.add_vline(x=model['mean_spend'], line_dash="dash", line_color="gray")
    fig.add_hline(y=model['mean_risk'], line_dash="dash", line_color="gray")
    event = st.plotly_chart(fig, on_select="rerun")

    # Details
    sel_id = event["selection"]["points"][0]["customdata"][0] if (event and event["selection"]["points"]) else (selected_supplier if selected_supplier != "All Suppliers" else None)
    if sel_id:
        if sel_id in agg_df['Supplier_ID'].values:
            data = agg_df[agg_df['Supplier_ID'] == sel_id].iloc[0]
            st.write(f"### Supplier: {data['Supplier_ID']}")
            st.metric("Spend", f"{data['Order Value USD']:,.0f} $")
            for r in RISK_COLS:
                st.write(f"**{r.replace('_Score', '').replace('_', ' ')}**: {data[r]:.2f}")
                st.progress(data[r])
        else:
            st.warning(f"Supplier {sel_id} has no orders in the selected period.")
    else:
        st.info("Select a point on the graph or a Supplier from the menu to see details.")

if __name__ == "__main__":
    main()