# # genetic_dashboard.py

# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.express as px
# from sklearn.preprocessing import LabelEncoder
# from sklearn.cluster import KMeans
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Page settings
# st.set_page_config(page_title="Genetic Risk Dashboard", layout="wide")

# st.title("🧬 Genetic Population Health Dashboard")

# # --- Load Data ---
# st.sidebar.header("Upload SNP Dataset")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     st.subheader("🔍 Raw Data Preview")
#     st.dataframe(df.head())

#     # --- Risk Level Distribution ---
#     st.subheader("📊 Disease Risk Distribution")
#     risk_counts = df['Risk'].value_counts().reset_index()
#     risk_counts.columns = ['Risk Level', 'Count']

#     risk_fig = px.pie(
#     risk_counts,
#     names='Risk Level',
#     values='Count',
#     title="Overall Disease Risk Levels",
#     hole=0.4
#     )
#     st.plotly_chart(risk_fig, use_container_width=True)


#     # --- Disease Frequency ---
#     st.subheader("📌 Disease Frequency by Risk Level")
#     disease_risk = df.groupby(['Disease', 'Risk']).size().reset_index(name='Count')
#     disease_fig = px.bar(disease_risk, x='Disease', y='Count',
#                          color='Risk', barmode='group',
#                          title="Disease Occurrence by Risk Level")
#     st.plotly_chart(disease_fig, use_container_width=True)

#     # --- SNP Frequency ---
#     st.subheader("🧬 SNP (rsID) Frequency")
#     snp_freq = df['rsID'].value_counts().reset_index()
#     snp_freq.columns = ['rsID', 'Count']
#     snp_fig = px.bar(snp_freq, x='rsID', y='Count',
#                      title="Frequency of SNPs in Dataset")
#     st.plotly_chart(snp_fig, use_container_width=True)

#     # --- Clustering Users by Genotype & Risk ---
#     st.subheader("🤖 Clustering Individuals by Genotype & Risk (KMeans)")

#     # Encode categorical features
#     df_cluster = df.copy()
#     for col in ['Genotype', 'Risk']:
#         df_cluster[col] = LabelEncoder().fit_transform(df_cluster[col])

#     # KMeans clustering
#     k = st.slider("Select number of clusters (K)", min_value=2, max_value=6, value=3)
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     df_cluster['Cluster'] = kmeans.fit_predict(df_cluster[['Genotype', 'Risk']])

#     cluster_fig = px.scatter(df_cluster, x='Genotype', y='Risk',
#                              color='Cluster', hover_data=['Disease'],
#                              title="KMeans Clustering of Individuals")
#     st.plotly_chart(cluster_fig, use_container_width=True)

#     # --- Symptom Word Cloud or List ---
#     st.subheader("🩺 Common Symptoms Breakdown")
#     all_symptoms = ', '.join(df['Symptoms'].astype(str).tolist()).split(',')
#     symptom_counts = pd.Series([s.strip().lower() for s in all_symptoms]).value_counts().reset_index()
#     symptom_counts.columns = ['Symptom', 'Count']
#     st.dataframe(symptom_counts)

# else:
#     st.warning("Please upload a dataset CSV to get started.")













# import streamlit as st
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="🧬 Genetic Risk Analyzer", layout="wide")
# st.title("🧬 Genetic Risk Analyzer Dashboard")

# # --- File Upload ---
# st.sidebar.header("📂 Upload Your Files")
# ref_file = st.sidebar.file_uploader("Upload SNP Reference File", type=["csv"])
# user_file = st.sidebar.file_uploader("Upload Genotype File (People)", type=["csv"])

# if ref_file and user_file:
#     # Read files
#     ref_df = pd.read_csv(ref_file)
#     user_df = pd.read_csv(user_file)
#     ref_rsids = set(ref_df['rsID'].unique())
#     user_rsids = set(user_df['rsID'].unique())
#     common_rsids = ref_rsids.intersection(user_rsids)
    
#     # Debug unique rsIDs
#     st.write("🔬 Unique rsIDs in Reference:", sorted(ref_df['rsID'].unique())[:10])
#     st.write("🔬 Unique rsIDs in User File:", sorted(user_df['rsID'].unique())[:10])

# # Check common rsIDs
#     ref_rsids = set(ref_df['rsID'].unique())
#     user_rsids = set(user_df['rsID'].unique())
#     common_rsids = ref_rsids.intersection(user_rsids)

#     st.write("✅ Total matching rsIDs:", len(common_rsids))

    
#     st.write("🧬 Total rsIDs in Reference File:", len(ref_rsids))
#     st.write("🧬 Total rsIDs in User File:", len(user_rsids))
#     st.write("✅ Common rsIDs found:", len(common_rsids))
#     st.write("🔁 Sample common rsIDs:", list(common_rsids)[:10])

    
#         # DEBUG: Show sample rsIDs and Genotypes before normalization
#     st.subheader("🔍 Debug Info")
#     st.write("📘 rsIDs in Reference File:", ref_df['rsID'].unique()[:10])
#     st.write("📗 rsIDs in User File:", user_df['rsID'].unique()[:10])
#     st.write("🧬 Sample Genotypes in Reference:", ref_df['Genotype'].unique()[:10])
#     st.write("🧬 Sample Genotypes in User File:", user_df['Genotype'].unique()[:10])
    
    


#     # Normalize genotypes: sort alleles alphabetically (e.g., CT == TC)
#     ref_df['Genotype'] = ref_df['Genotype'].apply(lambda g: ''.join(sorted(g)) if pd.notnull(g) else g)
#     # Remove separators like '/' or '|' from genotypes
#     user_df['Genotype'] = user_df['Genotype'].str.replace(r'[^ACGT]', '', regex=True)
#     ref_df['Genotype'] = ref_df['Genotype'].str.replace(r'[^ACGT]', '', regex=True)

#     user_df['Genotype'] = user_df['Genotype'].apply(lambda g: ''.join(sorted(g)) if pd.notnull(g) else g)

#     # Merge on rsID and Genotype
#     merged = pd.merge(user_df, ref_df, how='inner', on=['rsID', 'Genotype'])

#     st.success(f"Matched {len(merged)} entries between genotypes and SNP database.")

#     if len(merged) > 0:
#         # --- Sidebar: Select Person ---
#         person_ids = merged['PersonID'].unique().tolist()
#         selected_person = st.sidebar.selectbox("👤 Select a Person", person_ids)

#         # Filter for selected person
#         person_data = merged[merged['PersonID'] == selected_person]

#         # --- Display Raw Matches ---
#         st.subheader(f"🧬 Matched SNPs for {selected_person}")
#         st.dataframe(person_data[['rsID', 'Genotype', 'Disease', 'Risk', 'Symptoms']], use_container_width=True)

#         # --- Plot: Risk Distribution ---
#         risk_counts = person_data['Risk'].value_counts().reset_index()
#         risk_counts.columns = ['Risk', 'Count']
#         fig = px.pie(risk_counts, names='Risk', values='Count', title=f"📊 Risk Level Breakdown for {selected_person}")
#         st.plotly_chart(fig, use_container_width=True)

#         # --- Optional: Show All Possible Diseases ---
#         st.subheader("🩺 Possible Disease Risks")
#         for _, row in person_data.iterrows():
#             st.markdown(f"**{row['Disease']}**  ")
#             st.markdown(f"- 🧬 Genotype: `{row['Genotype']}` (rsID: `{row['rsID']}`)")
#             st.markdown(f"- ⚠️ Risk: `{row['Risk']}`")
#             st.markdown(f"- 💬 Symptoms: _{row['Symptoms']}_")
#             st.markdown("---")

#     else:
#         st.warning("No matching SNPs found. Check if rsID and Genotype formats are aligned.")

# else:
#     st.info("👈 Please upload both the SNP reference file and the genotype file to begin analysis.")


















# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# st.set_page_config(page_title="🧬 Genetic Risk Analyzer", layout="wide")
# st.title("🧬 Genetic Risk Analyzer Dashboard")

# # --- File Upload ---
# st.sidebar.header("📂 Upload Your Files")
# ref_file = st.sidebar.file_uploader("Upload SNP Reference File", type=["csv"])
# user_file = st.sidebar.file_uploader("Upload Genotype File (People)", type=["csv"])

# if ref_file and user_file:
#     # Read files
#     ref_df = pd.read_csv(ref_file)
#     user_df = pd.read_csv(user_file)

#     # Normalize genotypes: sort alleles alphabetically (e.g., CT == TC)
#     ref_df['Genotype'] = ref_df['Genotype'].apply(lambda g: ''.join(sorted(g)) if pd.notnull(g) else g)
#     user_df['Genotype'] = user_df['Genotype'].apply(lambda g: ''.join(sorted(g)) if pd.notnull(g) else g)

#     # Merge on rsID and Genotype
#     merged = pd.merge(user_df, ref_df, how='inner', on=['rsID', 'Genotype'])

#     st.success(f"Matched {len(merged)} entries between genotypes and SNP database.")

#     if len(merged) > 0:
#         # --- Sidebar: Select Person ---
#         person_ids = merged['PersonID'].unique().tolist()
#         selected_person = st.sidebar.selectbox("👤 Select a Person", person_ids)

#         # Filter for selected person
#         person_data = merged[merged['PersonID'] == selected_person]

#         # --- Display Raw Matches ---
#         st.subheader(f"🧬 Matched SNPs for {selected_person}")
#         st.dataframe(person_data[['rsID', 'Genotype', 'Disease', 'Risk', 'Symptoms']], use_container_width=True)

#         # --- Plot: Risk Distribution Pie ---
#         risk_counts = person_data['Risk'].value_counts().reset_index()
#         risk_counts.columns = ['Risk', 'Count']
#         fig = px.pie(risk_counts, names='Risk', values='Count', title=f"📊 Risk Level Breakdown for {selected_person}")
#         st.plotly_chart(fig, use_container_width=True)

#         # --- Bar Chart: Risk Level Frequency ---
#         st.subheader("📊 Risk Level Frequency (Bar Chart)")
#         bar_fig = px.bar(risk_counts, x='Risk', y='Count', color='Risk', title="Number of SNPs per Risk Level")
#         st.plotly_chart(bar_fig, use_container_width=True)

#         # --- Radar Chart: Disease Risk Profile ---
#         st.subheader("🕸️ Disease Risk Radar Chart")
#         radar_data = person_data.groupby('Disease')['Risk'].first().reset_index()
#         risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
#         radar_data['Risk_Score'] = radar_data['Risk'].map(risk_map)

#         radar_fig = go.Figure()
#         radar_fig.add_trace(go.Scatterpolar(
#             r=radar_data['Risk_Score'],
#             theta=radar_data['Disease'],
#             fill='toself',
#             name=selected_person
#         ))
#         radar_fig.update_layout(
#             polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
#             showlegend=False
#         )
#         st.plotly_chart(radar_fig, use_container_width=True)

#         # --- Optional: Show All Possible Diseases ---
#         st.subheader("🩺 Possible Disease Risks")
#         for _, row in person_data.iterrows():
#             st.markdown(f"**{row['Disease']}**  ")
#             st.markdown(f"- 🧬 Genotype: `{row['Genotype']}` (rsID: `{row['rsID']}`)")
#             st.markdown(f"- ⚠️ Risk: `{row['Risk']}`")
#             st.markdown(f"- 💬 Symptoms: _{row['Symptoms']}_")
#             st.markdown("---")

#     else:
#         st.warning("No matching SNPs found. Check if rsID and Genotype formats are aligned.")

# else:
#     st.info("👈 Please upload both the SNP reference file and the genotype file to begin analysis.")