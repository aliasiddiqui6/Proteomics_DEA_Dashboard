# Proteomics_DEA_Dashboard
End-to-end LC-MS proteomics differential expression pipeline and interactive dashboard

# Version a
This pipeline connects directly to the NCI Proteomic Data Commons via GraphQL. By providing a valid PDC Study ID, the script automatically downloads the relevant clinical and quantitation matrices, performs KNN missing value imputation, executes differential expression analysis, and serves the results to the dashboard.