


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

st.set_page_config(page_title="Genetic Risk Analyzer", layout="wide")
st.title(" Genetic Risk Analyzer Dashboard")

# --- File Upload ---
st.sidebar.header(" Upload Your Files")
ref_file = st.sidebar.file_uploader("Upload SNP Reference File", type=["csv"])
user_file = st.sidebar.file_uploader("Upload Genotype File (People)", type=["csv"])

def clean_data(df):
    df = df.drop_duplicates()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.dropna(subset=['rsID', 'Genotype'], inplace=True)
    df['Genotype'] = df['Genotype'].apply(lambda g: ''.join(sorted(g)) if pd.notnull(g) else g)
    return df

if ref_file and user_file:
    # Read & Clean Files
    ref_df = clean_data(pd.read_csv(ref_file))
    user_df = clean_data(pd.read_csv(user_file))

    # Merge Data
    merged = pd.merge(user_df, ref_df, how='inner', on=['rsID', 'Genotype'])
    st.success(f" Matched {len(merged)} SNPs between genotype and reference database.")

    if len(merged) > 0:
        # --- Sidebar: Select Person ---
        person_ids = merged['PersonID'].unique().tolist()
        selected_person = st.sidebar.selectbox(" Select a Person", person_ids)
        person_data = merged[merged['PersonID'] == selected_person]

        # Map Risk to Numeric
        risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
        merged['Risk_Score'] = merged['Risk'].map(risk_map)
        person_data['Risk_Score'] = person_data['Risk'].map(risk_map)

        st.subheader(f" Matched SNPs for {selected_person}")
        st.dataframe(person_data[['rsID', 'Genotype', 'Disease', 'Risk', 'Symptoms']], use_container_width=True)

        # --- Pie Chart
        risk_counts = person_data['Risk'].value_counts().reset_index()
        risk_counts.columns = ['Risk', 'Count']
        fig = px.pie(risk_counts, names='Risk', values='Count', title=f" Risk Level Breakdown for {selected_person}")
        st.plotly_chart(fig, use_container_width=True)

        # --- Bar Chart
        st.subheader(" Risk Level Frequency (Bar Chart)")
        bar_fig = px.bar(risk_counts, x='Risk', y='Count', color='Risk', title="Number of SNPs per Risk Level")
        st.plotly_chart(bar_fig, use_container_width=True)

        # --- Radar Chart
        st.subheader(" Disease Risk Radar Chart")
        radar_data = person_data.groupby('Disease')['Risk'].first().reset_index()
        radar_data['Risk_Score'] = radar_data['Risk'].map(risk_map)
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=radar_data['Risk_Score'],
            theta=radar_data['Disease'],
            fill='toself',
            name=selected_person
        ))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 3])), showlegend=False)
        st.plotly_chart(radar_fig, use_container_width=True)

        # --- Sunburst Chart
        st.subheader(" Sunburst Chart: Disease → Risk")
        sunburst_fig = px.sunburst(person_data, path=['Disease', 'Risk'], values='Risk_Score',
                                   color='Risk', title="Disease & Risk Hierarchy")
        st.plotly_chart(sunburst_fig, use_container_width=True)

        # --- Box Plot
        st.subheader(" Risk Score Spread per Disease")
        box_fig = px.box(person_data, x='Disease', y='Risk_Score', color='Risk',
                         title="Distribution of Risk Scores by Disease")
        st.plotly_chart(box_fig, use_container_width=True)

        # --- Histogram
        st.subheader(" Histogram of Risk Scores")
        hist_fig = px.histogram(person_data, x='Risk_Score', nbins=5, color='Risk',
                                title="Risk Score Distribution")
        st.plotly_chart(hist_fig, use_container_width=True)

        # --- Heatmap
        st.subheader(" Heatmap of Risk Scores per Disease")
        heatmap_df = person_data.pivot_table(index='Disease', columns='rsID', values='Risk_Score', fill_value=0)
        heatmap_fig = px.imshow(heatmap_df, aspect='auto', color_continuous_scale='Reds',
                                title="Disease vs SNP Risk Score Heatmap")
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # --- Sankey Diagram
        st.subheader(" SNP Flow: Genotype → Disease → Risk")
        all_nodes = list(pd.unique(person_data['Genotype'].tolist() +
                                   person_data['Disease'].tolist() +
                                   person_data['Risk'].tolist()))
        node_indices = {k: i for i, k in enumerate(all_nodes)}

        sankey_data = {
            "source": [],
            "target": [],
            "value": []
        }

        for row in person_data.itertuples():
            sankey_data["source"].append(node_indices[row.Genotype])
            sankey_data["target"].append(node_indices[row.Disease])
            sankey_data["value"].append(1)

        for row in person_data.itertuples():
            sankey_data["source"].append(node_indices[row.Disease])
            sankey_data["target"].append(node_indices[row.Risk])
            sankey_data["value"].append(1)

        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(label=all_nodes, pad=15, thickness=20),
            link=sankey_data
        )])
        sankey_fig.update_layout(title_text="Sankey Diagram of SNP → Disease → Risk", font_size=10)
        st.plotly_chart(sankey_fig, use_container_width=True)

        # --- Word Cloud of Symptoms
        st.subheader("Common Symptoms (Word Cloud)")
        symptoms_text = " ".join(person_data['Symptoms'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(symptoms_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

        # --- Average Risk per Disease
        st.subheader("Average Risk per Disease Across All Users")
        avg_risk = merged.groupby('Disease')['Risk_Score'].mean().reset_index().sort_values(by='Risk_Score', ascending=False)
        avg_risk_chart = px.bar(avg_risk, x='Disease', y='Risk_Score', color='Risk_Score',
                                title="Average Risk Score per Disease")
        st.plotly_chart(avg_risk_chart, use_container_width=True)

        # --- Genotype vs Disease Heatmap
        st.subheader("Genotype vs Disease Risk Heatmap")
        geno_disease_matrix = person_data.pivot_table(index='Genotype', columns='Disease', 
                                                      values='Risk_Score', aggfunc='mean', fill_value=0)
        geno_heatmap = px.imshow(geno_disease_matrix, aspect='auto', color_continuous_scale='Viridis',
                                 title="Average Risk Score per Genotype and Disease")
        st.plotly_chart(geno_heatmap, use_container_width=True)

        # --- Total Risk per Person
        st.subheader("Total Risk Score per Individual")
        total_risk = merged.groupby('PersonID')['Risk_Score'].sum().reset_index()
        total_risk_chart = px.bar(total_risk, x='PersonID', y='Risk_Score', title="Total Risk Score per Person")
        st.plotly_chart(total_risk_chart, use_container_width=True)

        # --- Markdown Output
        st.subheader(" Possible Disease Risks")
        for _, row in person_data.iterrows():
            st.markdown(f"**{row['Disease']}**")
            st.markdown(f"- Genotype: `{row['Genotype']}` (rsID: `{row['rsID']}`)")
            st.markdown(f"- Risk: `{row['Risk']}`")
            st.markdown(f"- Symptoms: _{row['Symptoms']}_")
            st.markdown("---")

    else:
        st.warning(" No matching SNPs found. Check if rsID and Genotype formats are aligned.")
else:
    st.info(" Please upload both the SNP reference file and the genotype file to begin analysis.")


















# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# st.set_page_config(page_title="Genetic Risk Analyzer", layout="wide")
# st.title(" Genetic Risk Analyzer Dashboard")

# # --- File Upload ---
# st.sidebar.header(" Upload Your Files")
# ref_file = st.sidebar.file_uploader("Upload SNP Reference File", type=["csv"])
# user_file = st.sidebar.file_uploader("Upload Genotype File (People)", type=["csv"])

# def clean_data(df):
#     df = df.drop_duplicates()
#     df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
#     df.dropna(subset=['rsID', 'Genotype'], inplace=True)
#     df['Genotype'] = df['Genotype'].apply(lambda g: ''.join(sorted(g)) if pd.notnull(g) else g)
#     return df

# if ref_file and user_file:
#     # Read & Clean Files
#     ref_df = clean_data(pd.read_csv(ref_file))
#     user_df = clean_data(pd.read_csv(user_file))

#     # Merge Data
#     merged = pd.merge(user_df, ref_df, how='inner', on=['rsID', 'Genotype'])
#     st.success(f" Matched {len(merged)} SNPs between genotype and reference database.")

#     if len(merged) > 0:
#         # --- Sidebar: Select Person ---
#         person_ids = merged['PersonID'].unique().tolist()
#         selected_person = st.sidebar.selectbox(" Select a Person", person_ids)
#         person_data = merged[merged['PersonID'] == selected_person]

#         # Map Risk to Numeric
#         risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
#         person_data['Risk_Score'] = person_data['Risk'].map(risk_map)

#         st.subheader(f" Matched SNPs for {selected_person}")
#         st.dataframe(person_data[['rsID', 'Genotype', 'Disease', 'Risk', 'Symptoms']], use_container_width=True)

#         # --- Pie Chart
#         risk_counts = person_data['Risk'].value_counts().reset_index()
#         risk_counts.columns = ['Risk', 'Count']
#         fig = px.pie(risk_counts, names='Risk', values='Count', title=f" Risk Level Breakdown for {selected_person}")
#         st.plotly_chart(fig, use_container_width=True)

#         # --- Bar Chart
#         st.subheader(" Risk Level Frequency (Bar Chart)")
#         bar_fig = px.bar(risk_counts, x='Risk', y='Count', color='Risk', title="Number of SNPs per Risk Level")
#         st.plotly_chart(bar_fig, use_container_width=True)

#         # --- Radar Chart
#         st.subheader(" Disease Risk Radar Chart")
#         radar_data = person_data.groupby('Disease')['Risk'].first().reset_index()
#         radar_data['Risk_Score'] = radar_data['Risk'].map(risk_map)
#         radar_fig = go.Figure()
#         radar_fig.add_trace(go.Scatterpolar(
#             r=radar_data['Risk_Score'],
#             theta=radar_data['Disease'],
#             fill='toself',
#             name=selected_person
#         ))
#         radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 3])), showlegend=False)
#         st.plotly_chart(radar_fig, use_container_width=True)

#         # --- Sunburst Chart
#         st.subheader(" Sunburst Chart: Disease → Risk")
#         sunburst_fig = px.sunburst(person_data, path=['Disease', 'Risk'], values='Risk_Score',
#                                    color='Risk', title="Disease & Risk Hierarchy")
#         st.plotly_chart(sunburst_fig, use_container_width=True)

#         # --- Box Plot
#         st.subheader(" Risk Score Spread per Disease")
#         box_fig = px.box(person_data, x='Disease', y='Risk_Score', color='Risk',
#                          title="Distribution of Risk Scores by Disease")
#         st.plotly_chart(box_fig, use_container_width=True)

#         # --- Histogram
#         st.subheader(" Histogram of Risk Scores")
#         hist_fig = px.histogram(person_data, x='Risk_Score', nbins=5, color='Risk',
#                                 title="Risk Score Distribution")
#         st.plotly_chart(hist_fig, use_container_width=True)

#         # --- Heatmap
#         st.subheader(" Heatmap of Risk Scores per Disease")
#         heatmap_df = person_data.pivot_table(index='Disease', columns='rsID', values='Risk_Score', fill_value=0)
#         heatmap_fig = px.imshow(heatmap_df, aspect='auto', color_continuous_scale='Reds',
#                                 title="Disease vs SNP Risk Score Heatmap")
#         st.plotly_chart(heatmap_fig, use_container_width=True)

#         # --- Sankey Diagram
#         st.subheader(" SNP Flow: Genotype → Disease → Risk")
#         all_nodes = list(pd.unique(person_data['Genotype'].tolist() +
#                                    person_data['Disease'].tolist() +
#                                    person_data['Risk'].tolist()))
#         node_indices = {k: i for i, k in enumerate(all_nodes)}

#         sankey_data = {
#             "source": [],
#             "target": [],
#             "value": []
#         }

#         for row in person_data.itertuples():
#             sankey_data["source"].append(node_indices[row.Genotype])
#             sankey_data["target"].append(node_indices[row.Disease])
#             sankey_data["value"].append(1)

#         for row in person_data.itertuples():
#             sankey_data["source"].append(node_indices[row.Disease])
#             sankey_data["target"].append(node_indices[row.Risk])
#             sankey_data["value"].append(1)

#         sankey_fig = go.Figure(data=[go.Sankey(
#             node=dict(label=all_nodes, pad=15, thickness=20),
#             link=sankey_data
#         )])
#         sankey_fig.update_layout(title_text="Sankey Diagram of SNP → Disease → Risk", font_size=10)
#         st.plotly_chart(sankey_fig, use_container_width=True)

#         # --- Markdown Output
#         st.subheader(" Possible Disease Risks")
#         for _, row in person_data.iterrows():
#             st.markdown(f"**{row['Disease']}**")
#             st.markdown(f"- Genotype: `{row['Genotype']}` (rsID: `{row['rsID']}`)")
#             st.markdown(f"- Risk: `{row['Risk']}`")
#             st.markdown(f"- Symptoms: _{row['Symptoms']}_")
#             st.markdown("---")

#     else:
#         st.warning(" No matching SNPs found. Check if rsID and Genotype formats are aligned.")
# else:
#     st.info(" Please upload both the SNP reference file and the genotype file to begin analysis.")

































# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # --- Streamlit Setup ---
# st.set_page_config(page_title=" Genetic Risk Analyzer", layout="wide")
# st.title("Genetic Risk Analyzer Dashboard")

# # --- File Upload ---
# st.sidebar.header("Upload Your Files")
# ref_file = st.sidebar.file_uploader("Upload SNP Reference File", type=["csv"])
# user_file = st.sidebar.file_uploader("Upload Genotype File (People)", type=["csv"])


# # --- Data Cleaning ---
# def clean_data(df):
#     df = df.drop_duplicates()
#     df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
#     df.dropna(subset=['rsID', 'Genotype'], inplace=True)
#     df['Genotype'] = df['Genotype'].apply(lambda g: ''.join(sorted(g)) if pd.notnull(g) else g)
#     return df


# # --- Visualization Functions ---
# def plot_pie_chart(risk_counts, title):
#     fig, ax = plt.subplots()
#     ax.pie(risk_counts['Count'], labels=risk_counts['Risk'], autopct='%1.1f%%', startangle=140)
#     ax.set_title(title)
#     return fig


# def plot_bar_chart(risk_counts):
#     fig, ax = plt.subplots()
#     sns.barplot(x='Risk', y='Count', data=risk_counts, hue='Risk', ax=ax)
#     ax.set_title("Number of SNPs per Risk Level")
#     return fig


# def plot_radar_chart(person_data, selected_person, risk_map):
#     radar_data = person_data.groupby('Disease')['Risk'].first().reset_index()
#     radar_data['Risk_Score'] = radar_data['Risk'].map(risk_map)
#     labels = radar_data['Disease'].values
#     stats = radar_data['Risk_Score'].values

#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     stats = np.concatenate((stats, [stats[0]]))
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#     ax.plot(angles, stats, 'o-', linewidth=2)
#     ax.fill(angles, stats, alpha=0.25)
#     ax.set_thetagrids(np.degrees(angles[:-1]), labels)
#     ax.set_title(f'Risk Radar Chart - {selected_person}')
#     ax.grid(True)
#     return fig


# def plot_boxplot(person_data):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.boxplot(x='Disease', y='Risk_Score', data=person_data, hue='Risk', ax=ax)
#     ax.set_title("Distribution of Risk Scores by Disease")
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#     return fig


# def plot_histogram(person_data):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.histplot(data=person_data, x='Risk_Score', bins=5, hue='Risk', multiple='stack', ax=ax)
#     ax.set_title("Risk Score Distribution")
#     return fig


# def plot_heatmap(person_data):
#     pivot = person_data.pivot_table(index='Disease', columns='rsID', values='Risk_Score', fill_value=0)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(pivot, cmap='Reds', linewidths=0.5, annot=False, ax=ax)
#     ax.set_title("Disease vs SNP Risk Score Heatmap")
#     return fig


# # --- Main Logic ---
# if ref_file and user_file:
#     ref_df = clean_data(pd.read_csv(ref_file))
#     user_df = clean_data(pd.read_csv(user_file))

#     merged = pd.merge(user_df, ref_df, how='inner', on=['rsID', 'Genotype'])
#     st.success(f"Matched {len(merged)} SNPs between genotype and reference database.")

#     if len(merged) > 0:
#         person_ids = merged['PersonID'].unique().tolist()
#         selected_person = st.sidebar.selectbox("Select a Person", person_ids)
#         person_data = merged[merged['PersonID'] == selected_person]

#         risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
#         person_data['Risk_Score'] = person_data['Risk'].map(risk_map)

#         st.subheader(f"Matched SNPs for {selected_person}")
#         st.dataframe(person_data[['rsID', 'Genotype', 'Disease', 'Risk', 'Symptoms']], use_container_width=True)

#         risk_counts = person_data['Risk'].value_counts().reset_index()
#         risk_counts.columns = ['Risk', 'Count']

#         st.subheader("Risk Level Breakdown (Pie Chart)")
#         st.pyplot(plot_pie_chart(risk_counts, f"Risk Level Breakdown for {selected_person}"))

#         st.subheader("Risk Level Frequency (Bar Chart)")
#         st.pyplot(plot_bar_chart(risk_counts))

#         st.subheader("Disease Risk Radar Chart")
#         st.pyplot(plot_radar_chart(person_data, selected_person, risk_map))

#         st.subheader("Risk Score Spread per Disease")
#         st.pyplot(plot_boxplot(person_data))

#         st.subheader("Histogram of Risk Scores")
#         st.pyplot(plot_histogram(person_data))

#         st.subheader("Heatmap of Risk Scores per Disease")
#         st.pyplot(plot_heatmap(person_data))

#         st.subheader("Possible Disease Risks")
#         for _, row in person_data.iterrows():
#             st.markdown(f"**{row['Disease']}**  ")
#             st.markdown(f"-  Genotype: `{row['Genotype']}` (rsID: `{row['rsID']}`)")
#             st.markdown(f"-  Risk: `{row['Risk']}`")
#             st.markdown(f"-  Symptoms: _{row['Symptoms']}_")
#             st.markdown("---")
#     else:
#         st.warning("No matching SNPs found. Check if rsID and Genotype formats are aligned.")
# else:
#     st.info("Please upload both the SNP reference file and the genotype file to begin analysis.")
