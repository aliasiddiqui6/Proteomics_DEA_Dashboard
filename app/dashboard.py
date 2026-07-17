import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
import io

st.set_page_config(
    page_title="PDC LC-MS Proteomics Dashboard", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def generate_mock_pdc_data(pdc_id):
    """
    Simulates fetching an abundance matrix and metadata from PDC.
    Creates a realistic dataset where a subset of proteins are intentionally
    differentially expressed between Tumor and Normal groups so PCA works realistically.
    """
    np.random.seed(42)
    n_proteins = 1000
    n_samples_per_group = 8
    
    proteins = [f"PROT_{str(i).zfill(4)}" for i in range(1, n_proteins + 1)]
    tumor_samples = [f"TUMOR_{i}" for i in range(1, n_samples_per_group + 1)]
    normal_samples = [f"NORMAL_{i}" for i in range(1, n_samples_per_group + 1)]
    all_samples = tumor_samples + normal_samples

    # Base expression matrix (Log2 transformed abundances, e.g., mean 15, std 2)
    matrix = np.random.normal(loc=15, scale=2, size=(n_proteins, len(all_samples)))
    df_expr = pd.DataFrame(matrix, index=proteins, columns=all_samples)
    
    # Introduce targeted biological signal (Differential Expression)
    # First 50 proteins: Upregulated in Tumor
    df_expr.iloc[0:50, 0:n_samples_per_group] += np.random.normal(loc=3, scale=1, size=(50, n_samples_per_group))
    # Next 50 proteins: Downregulated in Tumor
    df_expr.iloc[50:100, 0:n_samples_per_group] -= np.random.normal(loc=3, scale=1, size=(50, n_samples_per_group))

    # Metadata
    df_meta = pd.DataFrame({
        "Sample_ID": all_samples,
        "Group": ["Tumor"] * n_samples_per_group + ["Normal"] * n_samples_per_group
    })
    
    return df_expr, df_meta

@st.cache_data
def calculate_differential_expression(df_expr, df_meta, group1, group2):
    """
    Calculates Log2FC, p-values, and mock B-statistics comparing group1 vs group2.
    """
    g1_samples = df_meta[df_meta["Group"] == group1]["Sample_ID"].tolist()
    g2_samples = df_meta[df_meta["Group"] == group2]["Sample_ID"].tolist()

    # Calculate means
    g1_mean = df_expr[g1_samples].mean(axis=1)
    g2_mean = df_expr[g2_samples].mean(axis=1)
    
    # Log2 Fold Change
    log2fc = g1_mean - g2_mean
    
    # T-test (Welch's)
    t_stats, p_vals = stats.ttest_ind(df_expr[g1_samples], df_expr[g2_samples], axis=1, equal_var=False)
    
    # Handle NaNs from t-test (if variance is 0)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    t_stats = np.nan_to_num(t_stats, nan=0.0)

    # Simplified Benjamini-Hochberg FDR (Adj. P-Value)
    sorted_indices = np.argsort(p_vals)
    adj_p_vals = np.empty_like(p_vals)
    n = len(p_vals)
    for i, idx in enumerate(sorted_indices):
        adj_p_vals[idx] = min(1.0, p_vals[idx] * n / (i + 1))
        
    # Ensure monotonic increasing
    for i in range(n-2, -1, -1):
        idx = sorted_indices[i]
        next_idx = sorted_indices[i+1]
        adj_p_vals[idx] = min(adj_p_vals[idx], adj_p_vals[next_idx])

    # Mock B-statistic (log-odds of differential expression) based on t-stat magnitude
    b_stats = np.log(np.abs(t_stats) + 1e-5) * 2

    # Compile results
    results = pd.DataFrame({
        "Protein": df_expr.index,
        "Log2FC": log2fc.values,
        "t_stat": t_stats,
        "B_stat": b_stats,
        "P_Value": p_vals,
        "Adj_P_Value": adj_p_vals,
        "-Log10_P": -np.log10(p_vals + 1e-300) # prevent inf
    })
    return results

def get_csv_download(df):
    csv = df.to_csv(index=False)
    return csv.encode('utf-8')

def get_tsv_download(df):
    tsv = df.to_csv(index=False, sep='\t')
    return tsv.encode('utf-8')

st.sidebar.title("🧬 Pipeline Config")
st.sidebar.markdown("Configure PDC fetch and analysis parameters.")

pdc_id = st.sidebar.text_input("PDC Identifier", value="PDC000121", help="Enter the study ID from Proteomic Data Commons")

if pdc_id:
    # 1. Fetch & Parse Data
    try:
        df_expr, df_meta = generate_mock_pdc_data(pdc_id)
        available_groups = df_meta["Group"].unique().tolist()
        
        st.sidebar.success(f"Data Loaded: {df_expr.shape[0]} proteins, {df_expr.shape[1]} samples.")
        
        st.sidebar.subheader("Comparison Groups")
        group_case = st.sidebar.selectbox("Treatment / Case (Numerator)", available_groups, index=0)
        group_control = st.sidebar.selectbox("Control / Reference (Denominator)", available_groups, index=1 if len(available_groups)>1 else 0)
        
        st.sidebar.subheader("Significance Thresholds")
        fc_cutoff = st.sidebar.slider("Absolute Log2FC Threshold", 0.0, 5.0, 1.0, 0.1)
        pval_cutoff = st.sidebar.selectbox("Adjusted P-Value Threshold", [0.05, 0.01, 0.001], index=0)

        # 2. Run Statistics
        if group_case == group_control:
            st.error("Please select two different groups for comparison.")
        else:
            stats_df = calculate_differential_expression(df_expr, df_meta, group_case, group_control)
            
            # Label significance
            def get_status(row):
                if row["Adj_P_Value"] <= pval_cutoff and row["Log2FC"] >= fc_cutoff:
                    return "Up"
                elif row["Adj_P_Value"] <= pval_cutoff and row["Log2FC"] <= -fc_cutoff:
                    return "Down"
                return "NS"
                
            stats_df["Status"] = stats_df.apply(get_status, axis=1)
            up_count = len(stats_df[stats_df["Status"] == "Up"])
            down_count = len(stats_df[stats_df["Status"] == "Down"])
            ns_count = len(stats_df[stats_df["Status"] == "NS"])

            st.title(f"Proteomics Differential Expression: {pdc_id}")
            st.markdown(f"**Comparison:** `{group_case}` (vs) `{group_control}`")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Proteins Quantified", len(stats_df))
            col2.metric("🔺 Up-regulated DEPs", up_count)
            col3.metric("🔻 Down-regulated DEPs", down_count)

            tab_table, tab_volc, tab_dist, tab_pca = st.tabs([
                "📊 Stats Table", 
                "🌋 Volcano Plot", 
                "📈 Sample Distributions", 
                "🧬 PCA Analysis"
            ])

            with tab_table:
                st.subheader("Interactive Results Table")
                st.markdown("Filter, sort, or download the calculated differential expression metrics.")
                
                # Download buttons
                d_col1, d_col2 = st.columns([1, 10])
                with d_col1:
                    st.download_button(
                        label="Download CSV",
                        data=get_csv_download(stats_df),
                        file_name=f"{pdc_id}_DE_results.csv",
                        mime="text/csv",
                    )
                with d_col2:
                    st.download_button(
                        label="Download TSV",
                        data=get_tsv_download(stats_df),
                        file_name=f"{pdc_id}_DE_results.tsv",
                        mime="text/tab-separated-values",
                    )
                
                # Format dataframe for display
                display_df = stats_df.copy()
                # Format floats for cleaner viewing
                display_cols = ["Log2FC", "t_stat", "B_stat", "P_Value", "Adj_P_Value", "-Log10_P"]
                for col in display_cols:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.4g}" if x < 0.001 else f"{x:.4f}")
                
                # Stylize dataframe
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )

            with tab_volc:
                st.subheader("Volcano Plot")
                
                color_map = {"Up": "#EF553B", "Down": "#00CC96", "NS": "#B6E880"}
                
                fig_volc = px.scatter(
                    stats_df, 
                    x="Log2FC", 
                    y="-Log10_P", 
                    color="Status",
                    color_discrete_map=color_map,
                    hover_name="Protein",
                    hover_data={
                        "-Log10_P": False,
                        "Log2FC": ":.3f",
                        "Adj_P_Value": ":.3e"
                    },
                    template="plotly_white",
                    height=600
                )
                
                # Add threshold lines
                fig_volc.add_vline(x=fc_cutoff, line_dash="dash", line_color="grey")
                fig_volc.add_vline(x=-fc_cutoff, line_dash="dash", line_color="grey")
                # Estimate the -log10 P threshold (using raw p-value equivalent roughly, or just max adj P)
                # For visualization, we'll draw the line at the approximate -log10P of the cutoff
                approx_p_thresh = stats_df[stats_df["Adj_P_Value"] <= pval_cutoff]["P_Value"].max()
                if pd.notna(approx_p_thresh):
                    fig_volc.add_hline(y=-np.log10(approx_p_thresh), line_dash="dash", line_color="grey")

                fig_volc.update_layout(title="Volcano Plot (Significance vs Fold Change)")
                st.plotly_chart(fig_volc, use_container_width=True)

            with tab_dist:
                st.subheader("Abundance Distributions (Log2 Transformed)")
                dist_mode = st.radio("Select View:", ["Sample-wise Boxplots", "Group-wise Density"], horizontal=True)
                
                # Melt data for Plotly
                df_expr_reset = df_expr.reset_index()
                df_melt = pd.melt(df_expr_reset, id_vars=["index"], var_name="Sample_ID", value_name="Log2_Abundance")
                df_melt = df_melt.merge(df_meta, on="Sample_ID")

                if dist_mode == "Sample-wise Boxplots":
                    fig_box = px.box(
                        df_melt, 
                        x="Sample_ID", 
                        y="Log2_Abundance", 
                        color="Group",
                        template="plotly_white",
                        height=500
                    )
                    fig_box.update_layout(xaxis={'categoryorder': 'category ascending'})
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    fig_density = px.histogram(
                        df_melt, 
                        x="Log2_Abundance", 
                        color="Group", 
                        marginal="box",
                        histnorm='density',
                        barmode="overlay",
                        template="plotly_white",
                        height=500,
                        opacity=0.6
                    )
                    fig_density.update_layout(yaxis_title="Density")
                    st.plotly_chart(fig_density, use_container_width=True)

            with tab_pca:
                st.subheader("Principal Component Analysis (PCA)")
                
                pca_subset = st.selectbox(
                    "Select Features for PCA Calculation",
                    ["All Proteins", "All Significant DEPs", "Up-Regulated DEPs Only", "Down-Regulated DEPs Only"]
                )
                
                # Filter Expression Matrix based on selection
                proteins_to_use = stats_df["Protein"].tolist()
                
                if pca_subset == "All Significant DEPs":
                    proteins_to_use = stats_df[stats_df["Status"].isin(["Up", "Down"])]["Protein"].tolist()
                elif pca_subset == "Up-Regulated DEPs Only":
                    proteins_to_use = stats_df[stats_df["Status"] == "Up"]["Protein"].tolist()
                elif pca_subset == "Down-Regulated DEPs Only":
                    proteins_to_use = stats_df[stats_df["Status"] == "Down"]["Protein"].tolist()

                if len(proteins_to_use) < 2:
                    st.warning(f"Not enough features to calculate PCA ({len(proteins_to_use)} found). Adjust thresholds or select a different subset.")
                else:
                    try:
                        # Slice matrix, transpose so rows=samples, cols=features
                        matrix_pca = df_expr.loc[proteins_to_use].T
                        
                        # Calculate PCA
                        pca = PCA(n_components=2)
                        components = pca.fit_transform(matrix_pca)
                        
                        var_ratio = pca.explained_variance_ratio_ * 100
                        
                        # Create dataframe for plotting
                        df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
                        df_pca["Sample_ID"] = matrix_pca.index
                        df_pca = df_pca.merge(df_meta, on="Sample_ID")
                        
                        fig_pca = px.scatter(
                            df_pca, 
                            x="PC1", 
                            y="PC2", 
                            color="Group",
                            text="Sample_ID",
                            title=f"PCA Plot using {pca_subset} ({len(proteins_to_use)} features)",
                            labels={
                                "PC1": f"PC1 ({var_ratio[0]:.1f}% Variance)",
                                "PC2": f"PC2 ({var_ratio[1]:.1f}% Variance)"
                            },
                            template="plotly_white",
                            height=600
                        )
                        fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
                        st.plotly_chart(fig_pca, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error calculating PCA: {e}")

    except Exception as e:
        st.error(f"Failed to fetch or process data for ID {pdc_id}: {str(e)}")

else:
    st.info("👈 Please enter a PDC Identifier in the sidebar to begin.")