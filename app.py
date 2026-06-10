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
    st.title("Sustainable Supply Chain: Category-Specific Kraljic Matrix")
    df = load_data()
    models = train_category_models(df)

    # --- SIDEBAR ---
    selected_cat = st.sidebar.selectbox("Select Product Category", sorted(df['Product_Category'].unique()))
    selected_supplier = st.sidebar.selectbox("Select Supplier", ["All Suppliers"] + sorted(df["Supplier_ID"].unique().tolist()))
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", ["All Months"] + sorted(df['Month'].astype(str).unique().tolist()))
    
    if st.sidebar.button("Reset Selection"):
        st.rerun()

    if 'weights' not in st.session_state: st.session_state['weights'] = [20, 20, 20, 20, 20]
    st.sidebar.header("Risk Weights")
    new_weights = [st.sidebar.slider(col.replace('_Score', '').replace('_', ' '), 0, 100, st.session_state['weights'][i], 5) for i, col in enumerate(RISK_COLS)]
    st.session_state['weights'] = new_weights
    
    # --- PIPELINE ---
    subset = df[(df['Product_Category'] == selected_cat)]
    if selected_timeframe != "All Months": subset = subset[subset['Month'].astype(str) == selected_timeframe]
    
    agg_df = subset.groupby('Supplier_ID').agg({'Order Value USD': 'sum', 'Country': 'first', **{col: 'mean' for col in RISK_COLS}}).reset_index()
    
    if agg_df.empty:
        st.warning(f"No orders found for the selected category or period.")
        return

    model = models[selected_cat]
    agg_df['Normalized_Spend'] = np.clip(model['scaler'].transform(agg_df[['Order Value USD']]), 0, 1)
    agg_df['Weighted_Risk'] = (agg_df[RISK_COLS].values * (np.array(new_weights)/100)).sum(axis=1)
    agg_df['Kraljic_Quadrant'] = agg_df.apply(lambda x: get_kraljic_quadrant(x['Normalized_Spend'], x['Weighted_Risk'], model['mean_spend'], model['mean_risk']), axis=1)

    # --- PLOT ---
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

    # --- DETAILS ---
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

if __name__ == "__main__":
    main()