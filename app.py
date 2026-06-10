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
    'Standard Risk_Score', 
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
    models = {}
    for category in df['Product_Category'].unique():
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
    models = train_category_models(df)

    # --- SIDEBAR ---
    st.sidebar.header("Configuration")
    selected_cat = st.sidebar.selectbox("Select Product Category", sorted(df['Product_Category'].unique()))
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", ["All Months"] + sorted(df['Month'].unique().astype(str).tolist()))
    
    if st.sidebar.button("Reset Selection"):
        st.session_state['selected_point'] = None
        st.session_state['search_id'] = ""
        st.rerun()

    supplier_list = ["All Suppliers"] + sorted(df["Supplier_ID"].unique().tolist())

    selected_supplier = st.sidebar.selectbox(
        "Select Supplier",
        supplier_list
)
    
    # --- RISK WEIGHTS ---
    st.sidebar.header("Risk Weights")
    preset = st.sidebar.selectbox(
        "Weight Profile",
        [
            "Custom",
            "Balanced",
            "Quality Focus",
            "Financial Focus",
            "Sustainability Focus",
            "Political Focus"
        ]
    )

    if preset == "Balanced":
        st.session_state['weights'] = [20, 20, 20, 20, 20]

    elif preset == "Quality Focus":
        st.session_state['weights'] = [50, 10, 10, 10, 20]

    elif preset == "Financial Focus":
        st.session_state['weights'] = [10, 50, 10, 10, 20]

    elif preset == "Sustainability Focus":
        st.session_state['weights'] = [10, 10, 50, 20, 10]

    elif preset == "Political Focus":
        st.session_state['weights'] = [10, 10, 10, 20, 50]


    new_weights = []

    for i, col in enumerate(RISK_COLS):

        label = col.replace('_Score', '').replace('_', ' ')

        val = st.sidebar.slider(
            label,
            min_value=0,
            max_value=100,
            value=st.session_state['weights'][i],
            step=5
        )

        new_weights.append(val)

    st.session_state['weights'] = new_weights
    total_weight = sum(new_weights)

    if total_weight != 100:
        st.sidebar.error(
            f"⚠️ The sum of the weights must be 100! Current sum: {total_weight}"
        )
        st.stop()
    else:
        st.sidebar.success("✅ The sum of the weights is correct")

    weights = np.array(new_weights) / 100

    # --- DATA PIPELINE ---
    subset = df[df['Product_Category'] == selected_cat].copy()
    if selected_timeframe != "All Months":
        subset = subset[subset['Month'].astype(str) == selected_timeframe]
    
    agg_df = subset.groupby('Supplier_ID').agg(
        {'Order Value USD': 'sum', 'Country': 'first', **{col: 'mean' for col in RISK_COLS}}
    ).reset_index()
    
    model = models[selected_cat]
    norm_spend = np.clip(model['scaler'].transform(agg_df[['Order Value USD']]), 0, 1)
    weighted_risk = (agg_df[RISK_COLS].values * weights).sum(axis=1)
    
    agg_df['Normalized_Spend'] = norm_spend
    agg_df['Weighted_Risk'] = weighted_risk
    
    agg_df['Kraljic_Quadrant'] = agg_df.apply(
        lambda x: get_kraljic_quadrant(x['Normalized_Spend'], x['Weighted_Risk'], model['mean_spend'], model['mean_risk']), axis=1
    )

    # --- VISUALIZATION ---
    plot_df = agg_df.copy()
    if selected_supplier != "All Suppliers":
        plot_df = plot_df[
             plot_df["Supplier_ID"] == selected_supplier
    ]

    fig = px.scatter(
        plot_df, x="Normalized_Spend", y="Weighted_Risk", color="Kraljic_Quadrant",
        hover_data=['Supplier_ID'], range_x=[0, 1], range_y=[0, 1],
        category_orders={"Kraljic_Quadrant": ["Strategic", "Leverage", "Bottleneck", "Non-Critical"]}
    )
    fig.add_vline(x=model['mean_spend'], line_dash="dash", line_color="gray")
    fig.add_hline(y=model['mean_risk'], line_dash="dash", line_color="gray")
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='White')))
    fig.update_layout(height=600, width=900)
    
    event = st.plotly_chart(fig, on_select="rerun")
    
    if event and event["selection"]["points"]:
        st.session_state['selected_point'] = event["selection"]["points"][0]["customdata"][0]
    
    sel_id = st.session_state['selected_point']

    if selected_supplier != "All Suppliers":
        sel_id = selected_supplier

    # --- DETAILS ---
    if sel_id:
        data = agg_df[agg_df['Supplier_ID'] == sel_id].iloc[0]
        st.write(f"### Supplier: {data['Supplier_ID']}")
        st.write(f"**Country**: {data.get('Country', 'N/A')}") # Интегрировано
        st.metric("Spend", f"{data['Order Value USD']:,.0f} $")
        for r in RISK_COLS:
            score = data[r]
            st.write(f"**{r.replace('_Score', '').replace('_', ' ')}**: {score:.2f}")
            st.progress(score)
    else:
        st.info("Click a point or enter a Supplier ID to see details.")

if __name__ == "__main__":
    main()
