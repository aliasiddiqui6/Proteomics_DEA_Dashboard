import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Proteomics DGE Dashboard", layout="wide")
st.title("🧪 Proteomics Differential Expression Dashboard")
st.markdown("Explore significant protein changes between Tumor and Normal tissues.")

# 2. Sidebar Controls
st.sidebar.header("Filter Thresholds")
fc_threshold = st.sidebar.slider("Log2 Fold Change Threshold", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
p_threshold = st.sidebar.selectbox("P-value Threshold", options=[0.05, 0.01, 0.001], index=1)

# 3. Generate Dummy Data (To test the layout before using real data)
@st.cache_data # This caches the data so it doesn't reload on every slider move
def load_dummy_data():
    np.random.seed(42)
    proteins = [f"PROT_{i}" for i in range(1, 501)]
    logfc = np.random.normal(0, 2, 500)
    pvals = np.random.uniform(0, 0.1, 500)
    
    df = pd.DataFrame({"Protein": proteins, "Log2FC": logfc, "P_Value": pvals})
    df["-Log10_P"] = -np.log10(df["P_Value"])
    return df

df = load_dummy_data()

# 4. Apply Dynamic Filters based on Slider Inputs
# Categorize proteins based on thresholds
def categorize_significance(row):
    if row["P_Value"] < p_threshold and row["Log2FC"] >= fc_threshold:
        return "Upregulated"
    elif row["P_Value"] < p_threshold and row["Log2FC"] <= -fc_threshold:
        return "Downregulated"
    else:
        return "Not Significant"

df["Significance"] = df.apply(categorize_significance, axis=1)

# 5. Render the Volcano Plot
st.subheader("Volcano Plot")

# Map colors for the biological standard (Red=Up, Blue=Down, Grey=Not Sig)
color_map = {"Upregulated": "red", "Downregulated": "blue", "Not Significant": "lightgrey"}

fig = px.scatter(
    df, x="Log2FC", y="-Log10_P", 
    color="Significance", color_discrete_map=color_map,
    hover_name="Protein", hover_data={"P_Value": True},
    title=f"Thresholds: |Log2FC| > {fc_threshold}, p < {p_threshold}"
)

st.plotly_chart(fig, use_container_width=True)

# 6. Show the Filtered Data Table
st.subheader("Significant Proteins")
significant_df = df[df["Significance"] != "Not Significant"]
st.dataframe(significant_df)