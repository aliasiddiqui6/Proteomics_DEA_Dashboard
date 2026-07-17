import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.stats.multitest as mt
from sklearn.decomposition import PCA

st.set_page_config(page_title="Proteomics DGE Dashboard", layout="wide")
st.title("🔬 High-Throughput Proteomics DGE Pipeline")

# --- 1. SIDEBAR: DATA INPUT & THRESHOLDS ---
st.sidebar.header("1. Data Fetching")
pdc_id = st.sidebar.text_input("Enter PDC Identifier", value="PDC000121")
fetch_btn = st.sidebar.button("Fetch Data")

st.sidebar.header("2. Analysis Thresholds")
fc_threshold = st.sidebar.slider("Log2 Fold Change Threshold", 0.5, 3.0, 1.0, 0.1)
p_threshold = st.sidebar.selectbox("P-value Threshold", [0.05, 0.01, 0.001], index=0)

# --- MOCK DATA GENERATOR (Simulating the API Fetch) ---
@st.cache_data
def load_mock_matrix(pdc_id):
    np.random.seed(42)
    proteins = [f"PROT_{i}" for i in range(1, 1001)]
    # Simulate 5 Tumor and 5 Normal samples
    tumor_data = np.random.normal(loc=15, scale=2, size=(1000, 5))
    normal_data = np.random.normal(loc=14, scale=2, size=(1000, 5))
    
    # Introduce artificial DE for the first 100 proteins
    tumor_data[:50, :] += 3  # Upregulated
    normal_data[50:100, :] += 3 # Downregulated
    
    columns = [f"Tumor_S{i}" for i in range(1, 6)] + [f"Normal_S{i}" for i in range(1, 6)]
    df_raw = pd.DataFrame(np.hstack((tumor_data, normal_data)), columns=columns, index=proteins)
    
    # Extract metadata/groups based on column names
    groups = ["Tumor" if "Tumor" in col else "Normal" for col in columns]
    return df_raw, groups

# Helper function to apply universal white background styling
def apply_white_theme(fig):
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black')
    )
    return fig

# --- 2. MAIN APPLICATION LOGIC ---
if fetch_btn or 'df_raw' in st.session_state:
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw, st.session_state.groups = load_mock_matrix(pdc_id)
    
    df_raw = st.session_state.df_raw
    groups = st.session_state.groups
    
    st.success(f"Data loaded for {pdc_id}. Identified {df_raw.shape[1]} samples and {df_raw.shape[0]} proteins.")
    unique_groups = list(set(groups))
    
    col1, col2 = st.columns(2)
    with col1:
        group_a = st.selectbox("Select Experimental Group (Numerator)", unique_groups, index=0)
    with col2:
        group_b = st.selectbox("Select Control Group (Denominator)", unique_groups, index=1)

    if group_a != group_b:
        # --- 3. MATHEMATICAL ENGINE (DE CALCULATION) ---
        cols_a = [df_raw.columns[i] for i, g in enumerate(groups) if g == group_a]
        cols_b = [df_raw.columns[i] for i, g in enumerate(groups) if g == group_b]
        
        data_a = df_raw[cols_a]
        data_b = df_raw[cols_b]
        
        results = pd.DataFrame(index=df_raw.index)
        results['Mean_A'] = data_a.mean(axis=1)
        results['Mean_B'] = data_b.mean(axis=1)
        results['Log2FC'] = results['Mean_A'] - results['Mean_B']
        
        t_stat, p_val = stats.ttest_ind(data_a, data_b, axis=1)
        results['t_stat'] = t_stat
        results['p_value'] = p_val
        results['B_stat'] = results['t_stat'] ** 2 
        
        results = results.dropna()
        _, adj_p_val, _, _ = mt.multipletests(results['p_value'], method='fdr_bh')
        results['adj_p_value'] = adj_p_val
        results['-Log10_P'] = -np.log10(results['p_value'])
        
        conditions = [
            (results['p_value'] < p_threshold) & (results['Log2FC'] >= fc_threshold),
            (results['p_value'] < p_threshold) & (results['Log2FC'] <= -fc_threshold)
        ]
        choices = ['Upregulated', 'Downregulated']
        results['Significance'] = np.select(conditions, choices, default='Not Significant')
        
        up_count = (results['Significance'] == 'Upregulated').sum()
        down_count = (results['Significance'] == 'Downregulated').sum()
        
        # --- 4. TABBED VISUALIZATION INTERFACE ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "Results Table", 
            "Volcano Plot", 
            "Distributions", 
            "PCA"
        ])
        
        with tab1:
            st.subheader(f"Differential Expression Results: {group_a} vs {group_b}")
            st.markdown(f"**Upregulated:** <span style='color:red'>{up_count}</span> | **Downregulated:** <span style='color:blue'>{down_count}</span>", unsafe_allow_html=True)
            st.dataframe(results.sort_values('p_value'))
            
            # Download buttons side-by-side and equal width
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv = results.to_csv().encode('utf-8')
                st.download_button("Download Full Results (CSV)", csv, "DE_results.csv", "text/csv", use_container_width=True)
            with col_dl2:
                tsv = results.to_csv(sep='\t').encode('utf-8')
                st.download_button("Download Full Results (TSV)", tsv, "DE_results.tsv", "text/tab-separated-values", use_container_width=True)
            
        with tab2:
            st.subheader("Volcano Plot")
            color_map = {"Upregulated": "red", "Downregulated": "blue", "Not Significant": "lightgrey"}
            fig_volc = px.scatter(
                results.reset_index(), x='Log2FC', y='-Log10_P', color='Significance',
                color_discrete_map=color_map, hover_name='index',
                hover_data={'p_value': True, 'adj_p_value': True},
                title=f"Volcano Plot ({group_a} vs {group_b})"
            )
            fig_volc.add_vline(x=fc_threshold, line_dash="dash", line_color="black", line_width=1)
            fig_volc.add_vline(x=-fc_threshold, line_dash="dash", line_color="black", line_width=1)
            fig_volc.add_hline(y=-np.log10(p_threshold), line_dash="dash", line_color="black", line_width=1)
            
            fig_volc = apply_white_theme(fig_volc)
            fig_volc.update_xaxes(showgrid=True, gridcolor='whitesmoke', linecolor='black')
            fig_volc.update_yaxes(showgrid=True, gridcolor='whitesmoke', linecolor='black')
            st.plotly_chart(fig_volc, use_container_width=True)
            
        with tab3:
            st.subheader("Sample-wise and Group-wise Distributions")
            # Group colors avoiding red/blue to prevent confusion
            group_color_map = {group_a: "#1b9e77", group_b: "#d95f02"} 
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                df_melted = df_raw.melt(var_name="Sample", value_name="Log2 Intensity")
                df_melted['Group'] = df_melted['Sample'].apply(lambda x: group_a if group_a in x else group_b)
                fig_box = px.box(df_melted, x="Sample", y="Log2 Intensity", color="Group", 
                                 color_discrete_map=group_color_map, title="Sample-wise Box Plot")
                fig_box = apply_white_theme(fig_box)
                fig_box.update_xaxes(linecolor='black')
                fig_box.update_yaxes(linecolor='black')
                st.plotly_chart(fig_box, use_container_width=True)
                
            with col_dist2:
                fig_dens = go.Figure()
                for group_name in unique_groups:
                    group_cols = [col for col, g in zip(df_raw.columns, groups) if g == group_name]
                    group_vals = df_raw[group_cols].values.flatten()
                    fig_dens.add_trace(go.Violin(x=group_vals, name=group_name, side='positive', 
                                                 line_color=group_color_map.get(group_name, "gray")))
                fig_dens.update_layout(title="Group-wise Density Distribution", xaxis_title="Log2 Intensity")
                fig_dens = apply_white_theme(fig_dens)
                fig_dens.update_xaxes(linecolor='black')
                fig_dens.update_yaxes(linecolor='black')
                st.plotly_chart(fig_dens, use_container_width=True)
                
        with tab4:
            st.subheader("Principal Component Analysis (PCA)")
            pca_filter = st.radio("Select Proteins for PCA:", ["All Proteins", "All DEPs", "Only Upregulated", "Only Downregulated"], horizontal=True)
            
            if pca_filter == "All DEPs":
                target_proteins = results[results['Significance'] != 'Not Significant'].index
            elif pca_filter == "Only Upregulated":
                target_proteins = results[results['Significance'] == 'Upregulated'].index
            elif pca_filter == "Only Downregulated":
                target_proteins = results[results['Significance'] == 'Downregulated'].index
            else:
                target_proteins = results.index
                
            if len(target_proteins) < 3:
                st.warning("Not enough proteins to perform PCA with current thresholds.")
            else:
                pca_data = df_raw.loc[target_proteins].T
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(pca_data)
                
                pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                pca_df['Sample'] = pca_data.index
                pca_df['Group'] = groups
                
                var_exp = pca.explained_variance_ratio_
                
                # Publication-ready PCA styling
                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Group', text='Sample',
                    color_discrete_map=group_color_map,
                    title=f"PCA Plot ({pca_filter} - {len(target_proteins)} features)",
                    labels={'PC1': f"PC1 ({var_exp[0]*100:.1f}%)", 'PC2': f"PC2 ({var_exp[1]*100:.1f}%)"}
                )
                
                fig_pca.update_traces(
                    textposition='top center', 
                    marker=dict(size=14, line=dict(width=1.5, color='black')),
                    textfont=dict(color='black', size=11)
                )
                
                fig_pca.update_layout(
                    template='simple_white', # Removes gridlines entirely
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='black', size=14, family="Arial"),
                    title=dict(font=dict(size=18)),
                    legend=dict(title_font_family="Arial", bordercolor="black", borderwidth=1)
                )
                
                # Thick bounding box axes (mirror=True creates the box effect)
                fig_pca.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')
                fig_pca.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')
                
                st.plotly_chart(fig_pca, use_container_width=True)