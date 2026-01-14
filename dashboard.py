import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from platform_helper import create_platform_analysis

# Page config
st.set_page_config(page_title='Survey Demographics Dashboard', layout='wide')

# Load data
@st.cache_data
def load_data():
    # Load cleaned FY26 data from /clean folder
    df_fy26 = pd.read_csv('clean/CLEAN_FY26.csv')

    # Load cleaned FY25Q4 data from /clean folder (already in wide format with remapped questions)
    df_fy25 = pd.read_csv('clean/CLEAN_FY25Q4.csv')

    # Load answer mapping
    questions_df = pd.read_csv('clean/CLEAN_FY26_ANSWER.csv')

    return df_fy26, df_fy25, questions_df

# Helper function to create filter section with cascading township
def create_filters(df, key_prefix):
    """Create filter controls with cascading Region -> Township"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        digital_filter = st.selectbox('Digital/Direct', ['ALL', 'DIGITAL', 'DIRECT'], key=f'{key_prefix}_dd')

    with col2:
        gender_filter = st.selectbox('Gender', ['All', 'Male', 'Female'], key=f'{key_prefix}_gender')

    with col3:
        age_filter = st.selectbox('Age Group', ['All', 'Under 20', '20-30', '31-40', '41-50', '51-60', 'Over 60'], key=f'{key_prefix}_age')

    with col4:
        all_regions = ['All'] + sorted(df['Region'].unique().tolist())
        region_filter = st.selectbox('Region', all_regions, key=f'{key_prefix}_region')

    with col5:
        # Cascading Township filter based on Region selection
        if region_filter != 'All':
            townships = ['All'] + sorted(df[df['Region'] == region_filter]['Township'].unique().tolist())
        else:
            townships = ['All'] + sorted(df['Township'].unique().tolist())
        township_filter = st.selectbox('Township', townships, key=f'{key_prefix}_township')

    return {
        'digital': digital_filter,
        'gender': gender_filter,
        'age': age_filter,
        'region': region_filter,
        'township': township_filter
    }

# Helper function to apply filters to dataframe
def apply_filters(df, filters):
    """Apply filters to dataframe"""
    df_filtered = df.copy()

    if filters['digital'] != 'ALL':
        df_filtered = df_filtered[df_filtered['Direct or Digital?'] == filters['digital'].title()]

    if filters['gender'] != 'All' and 'Q03' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Q03'] == filters['gender']]

    if filters['age'] != 'All' and 'Q02' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Q02'] == filters['age']]

    if filters['region'] != 'All':
        df_filtered = df_filtered[df_filtered['Region'] == filters['region']]

    if filters['township'] != 'All':
        df_filtered = df_filtered[df_filtered['Township'] == filters['township']]

    return df_filtered

df, df_fy25, questions_df = load_data()

# Title
st.title('📊 Survey Demographics Dashboard')

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Demographics', 'Digital Behavior & Engagement', 'Internet Usage Behaviors', 'Brand & Product Awareness', 'Location Impact Analysis'])

with tab1:
    # Demographics metrics
    st.header('Demographics Overview')

    # Create filters for Demographics
    demo_filters = create_filters(df, 'demo')

    # Filter data based on selections for Demographics
    df_demo = apply_filters(df, demo_filters)

    # Display sample size at the top
    st.markdown(f"**Sample Size: n = {len(df_demo)}**")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    # Total respondents
    col1.metric('Total Respondents', len(df_demo))

    # Direct percentage
    if 'Direct or Digital?' in df_demo.columns:
        direct_count = (df_demo['Direct or Digital?'] == 'Direct').sum()
        direct_pct = (direct_count / len(df_demo) * 100) if len(df_demo) > 0 else 0
        col2.metric('Direct', f"{direct_pct:.1f}%")
    else:
        col2.metric('Direct', 'N/A')

    # Male percentage
    if 'Q03' in df_demo.columns:
        male_count = (df_demo['Q03'] == 'Male').sum()
        male_pct = (male_count / len(df_demo) * 100) if len(df_demo) > 0 else 0
        col3.metric('Male', f"{male_pct:.1f}%")
    else:
        col3.metric('Male', 'N/A')

    # Heard of Po Chat percentage
    if 'Q31_g' in df_demo.columns:
        po_chat_count = (df_demo['Q31_g'] == 1.0).sum()
        po_chat_pct = (po_chat_count / len(df_demo) * 100) if len(df_demo) > 0 else 0
        col4.metric('Heard Po Chat', f"{po_chat_pct:.1f}%")
    else:
        col4.metric('Heard Po Chat', 'N/A')

    # Average NPS Score
    if 'Q40' in df_demo.columns:
        nps_avg = df_demo['Q40'].mean()
        col5.metric('Avg NPS Score', f"{nps_avg:.2f}")
    else:
        col5.metric('Avg NPS Score', 'N/A')

    # Apply same filters to FY25 data for Demographics tab
    fy25_demo_filtered = apply_filters(df_fy25, demo_filters)

    # First row: Regional Distribution (pie) and Age Distribution (bar)
    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:
        st.subheader('Regional Distribution')
        demo_col1a, demo_col1b = st.columns(2)

        with demo_col1a:
            st.markdown('**FY25**')
            # FY25 uses Q04 for region, FY26 uses 'Region'
            fy25_region_col = 'Q04' if 'Q04' in fy25_demo_filtered.columns else 'Region'
            if fy25_region_col in fy25_demo_filtered.columns:
                region_counts_fy25 = fy25_demo_filtered[fy25_region_col].value_counts().reset_index()
                region_counts_fy25.columns = ['Region', 'Count']

                fig_region_fy25 = px.pie(
                    region_counts_fy25,
                    names='Region',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_region_fy25.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_region_fy25.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_region_fy25, use_container_width=True, key='region_pie_chart_fy25')

        with demo_col1b:
            st.markdown('**FY26**')
            region_counts = df_demo['Region'].value_counts().reset_index()
            region_counts.columns = ['Region', 'Count']

            fig_region = px.pie(
                region_counts,
                names='Region',
                values='Count',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_region.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
            fig_region.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_region, use_container_width=True, key='region_pie_chart')

    with demo_col2:
        st.subheader('Age Distribution')
        age_order = ['Under 20', '20-30', '31-40', '41-50', '51-60', 'Over 60']

        if 'Q02' in df_demo.columns:
            # FY26 data
            age_counts_fy26 = df_demo['Q02'].value_counts().reset_index()
            age_counts_fy26.columns = ['Age Group', 'Count']
            age_counts_fy26['Age Group'] = pd.Categorical(age_counts_fy26['Age Group'], categories=age_order, ordered=True)
            age_counts_fy26 = age_counts_fy26.sort_values('Age Group')
            age_counts_fy26['Percentage'] = (age_counts_fy26['Count'] / age_counts_fy26['Count'].sum() * 100).round(1)

            # FY25 data
            age_counts_fy25 = fy25_demo_filtered['Q02'].value_counts().reset_index() if 'Q02' in fy25_demo_filtered.columns else pd.DataFrame()
            if len(age_counts_fy25) > 0:
                age_counts_fy25.columns = ['Age Group', 'Count']
                age_counts_fy25['Age Group'] = pd.Categorical(age_counts_fy25['Age Group'], categories=age_order, ordered=True)
                age_counts_fy25 = age_counts_fy25.sort_values('Age Group')
                age_counts_fy25['Percentage'] = (age_counts_fy25['Count'] / age_counts_fy25['Count'].sum() * 100).round(1)

            fig_age = go.Figure()

            # FY26 bars (blue)
            fig_age.add_trace(go.Bar(
                x=age_counts_fy26['Age Group'],
                y=age_counts_fy26['Percentage'],
                name='FY26',
                marker_color='#4575b4',
                text=[f'{p:.1f}%' for p in age_counts_fy26['Percentage']],
                textposition='outside',
                hovertemplate='FY26<br>%{x}<br>%{y:.1f}%<extra></extra>'
            ))

            # FY25 line (red)
            if len(age_counts_fy25) > 0:
                fig_age.add_trace(go.Scatter(
                    x=age_counts_fy25['Age Group'],
                    y=age_counts_fy25['Percentage'],
                    name='FY25',
                    mode='lines+markers',
                    line=dict(color='#d73027', width=3, dash='dash'),
                    marker=dict(size=10, color='#d73027'),
                    hovertemplate='FY25<br>%{x}<br>%{y:.1f}%<extra></extra>'
                ))

            fig_age.update_layout(
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis_title='',
                yaxis_title='Percentage (%)',
                xaxis=dict(categoryorder='array', categoryarray=age_order)
            )
            st.plotly_chart(fig_age, use_container_width=True, key='age_bar_chart')

    # Second row: Sources of Farming Information and Mobile Top-up Amount
    demo_col3, demo_col4 = st.columns(2)

    with demo_col3:
        st.subheader('Sources of Farming Information')
        q09_labels = {
            'Q09_a': 'Friend farmers',
            'Q09_b': 'Social media',
            'Q09_c': 'Input shops',
            'Q09_d': 'Yetagon/Proximity',
            'Q09_e': 'Other orgs',
            'Q09_f': 'Others'
        }

        # Prepare data for UpSet plot - FY26
        q09_cols = [col for col in q09_labels.keys() if col in df_demo.columns]

        # Create a DataFrame for upset plot - need to convert to boolean
        upset_data = df_demo[q09_cols].copy()
        upset_data = upset_data.fillna(0)
        upset_data = (upset_data == 1.0)

        # Rename columns to friendly names
        rename_map = {col: q09_labels[col] for col in q09_cols if col in q09_labels}
        upset_data = upset_data.rename(columns=rename_map)

        # Count combinations for FY26
        from collections import Counter
        combinations_fy26 = []
        for idx, row in upset_data.iterrows():
            combo = tuple(sorted(col for col in upset_data.columns if row[col]))
            if combo:
                combinations_fy26.append(combo)

        combo_counts_fy26 = Counter(combinations_fy26)

        # Prepare FY25 data for the same combinations
        q09_cols_fy25 = [col for col in q09_labels.keys() if col in fy25_demo_filtered.columns]
        upset_data_fy25 = fy25_demo_filtered[q09_cols_fy25].copy()
        upset_data_fy25 = upset_data_fy25.fillna(0)
        upset_data_fy25 = (upset_data_fy25 == 1.0)
        upset_data_fy25 = upset_data_fy25.rename(columns=rename_map)

        combinations_fy25 = []
        for idx, row in upset_data_fy25.iterrows():
            combo = tuple(sorted(col for col in upset_data_fy25.columns if row[col]))
            if combo:
                combinations_fy25.append(combo)

        combo_counts_fy25 = Counter(combinations_fy25)

        # Get top 15 combinations from FY26
        top_combos = sorted(combo_counts_fy26.items(), key=lambda x: x[1], reverse=True)[:15]

        # Create upset-style visualization manually
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Prepare data
        combo_labels = []
        combo_percentages_fy26 = []
        combo_percentages_fy25 = []
        matrix_data = {col: [] for col in upset_data.columns}

        total_respondents_fy26 = len(df_demo)
        total_respondents_fy25 = len(fy25_demo_filtered)

        for combo, count in top_combos:
            combo_labels.append(' & '.join(combo) if len(combo) <= 2 else f"{len(combo)} sources")
            pct_fy26 = (count / total_respondents_fy26 * 100) if total_respondents_fy26 > 0 else 0
            combo_percentages_fy26.append(pct_fy26)

            # Get FY25 count for the same combination
            count_fy25 = combo_counts_fy25.get(combo, 0)
            pct_fy25 = (count_fy25 / total_respondents_fy25 * 100) if total_respondents_fy25 > 0 else 0
            combo_percentages_fy25.append(pct_fy25)

            for col in upset_data.columns:
                matrix_data[col].append(1 if col in combo else 0)

        # Create figure with subplots
        fig_q09 = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.1,
            specs=[[{"type": "bar"}], [{"type": "scatter"}]]
        )

        # Top plot: bar chart of combination percentages (FY26)
        fig_q09.add_trace(
            go.Bar(
                x=list(range(len(combo_percentages_fy26))),
                y=combo_percentages_fy26,
                marker_color='#4575b4',
                name='FY26',
                width=0.6,
                text=[f'{p:.1f}%' for p in combo_percentages_fy26],
                textposition='outside',
                hovertemplate='FY26<br>%{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Overlay FY25 line on top of bars
        fig_q09.add_trace(
            go.Scatter(
                x=list(range(len(combo_percentages_fy25))),
                y=combo_percentages_fy25,
                mode='lines+markers',
                name='FY25',
                line=dict(color='#d73027', width=2, dash='dash'),
                marker=dict(size=8, color='#d73027'),
                hovertemplate='FY25<br>%{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Bottom plot: matrix showing which sources are in each combination
        # Add connecting lines first
        for j in range(len(matrix_data[list(upset_data.columns)[0]])):
            active_indices = [i for i, col in enumerate(upset_data.columns) if matrix_data[col][j] == 1]

            if len(active_indices) > 1:
                fig_q09.add_trace(
                    go.Scatter(
                        x=[j] * len(active_indices),
                        y=active_indices,
                        mode='lines',
                        line=dict(color='#4575b4', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=2, col=1
                )

        # Add dots for active sources
        for i, col in enumerate(upset_data.columns):
            x_vals = []
            y_vals = []

            for j in range(len(matrix_data[col])):
                if matrix_data[col][j] == 1:
                    x_vals.append(j)
                    y_vals.append(i)

            fig_q09.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    marker=dict(size=10, color='#4575b4', line=dict(color='#ffffff', width=1)),
                    showlegend=False,
                    name=col,
                    hovertemplate=f'{col}<extra></extra>'
                ),
                row=2, col=1
            )

        # Update layout
        fig_q09.update_xaxes(
            showticklabels=False,
            showgrid=False,
            range=[-0.5, len(combo_percentages_fy26) - 0.5],
            row=1, col=1
        )
        fig_q09.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, len(combo_percentages_fy26) - 0.5],
            row=2, col=1
        )
        fig_q09.update_yaxes(
            title_text="Percentage (%)",
            showgrid=True,
            gridcolor='#e0e0e0',
            row=1, col=1
        )
        fig_q09.update_yaxes(
            ticktext=list(upset_data.columns),
            tickvals=list(range(len(upset_data.columns))),
            showgrid=False,
            zeroline=False,
            row=2, col=1
        )

        fig_q09.update_layout(
            height=500,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_q09, use_container_width=True, key='farming_info_upset')

    with demo_col4:
        st.subheader('Mobile Top-up Amount per Week')

        # Standardize labels
        label_mapping = {
            'Less than 3000 MMK': '<3000 MMK',
            'less than 3000 MMK': '<3000 MMK',
            'approximately 5000 MMK': '~5000 MMK',
            'above 10000 MMK': '>10000 MMK',
            'other': 'Other'
        }

        demo_col4a, demo_col4b = st.columns(2)

        with demo_col4a:
            st.markdown('**FY25**')
            if 'Q29' in fy25_demo_filtered.columns:
                q29_counts_fy25 = fy25_demo_filtered['Q29'].value_counts().reset_index()
                q29_counts_fy25.columns = ['Top-up Amount', 'Count']
                q29_counts_fy25['Top-up Amount'] = q29_counts_fy25['Top-up Amount'].map(label_mapping).fillna(q29_counts_fy25['Top-up Amount'])

                fig_q29_fy25 = px.pie(
                    q29_counts_fy25,
                    names='Top-up Amount',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_q29_fy25.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_q29_fy25.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_q29_fy25, use_container_width=True, key='topup_chart_fy25')

        with demo_col4b:
            st.markdown('**FY26**')
            if 'Q29' in df_demo.columns:
                q29_counts = df_demo['Q29'].value_counts().reset_index()
                q29_counts.columns = ['Top-up Amount', 'Count']
                q29_counts['Top-up Amount'] = q29_counts['Top-up Amount'].map(label_mapping).fillna(q29_counts['Top-up Amount'])

                fig_q29 = px.pie(
                    q29_counts,
                    names='Top-up Amount',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_q29.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_q29.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_q29, use_container_width=True, key='topup_chart')

    # Third row: Types of Crops Grown (Co-occurrence Heatmap)
    st.subheader('Crop Co-occurrence Matrix')
    st.caption('Diagonal shows percentage of farmers growing each crop. Off-diagonal shows percentage of farmers growing both crops. Right heatmap shows FY26 - FY25 difference (green = increase, purple = decrease).')

    q06_labels = {
        'Q06_a': 'Rice',
        'Q06_b': 'Sesame',
        'Q06_c': 'Groundnut',
        'Q06_d': 'Pulses',
        'Q06_e': 'Vegetables',
        'Q06_f': 'Flowers',
        'Q06_g': 'Long-term trees',
        'Q06_h': 'Chili',
        'Q06_i': 'Tomato',
        'Q06_j': 'Onion',
        'Q06_k': 'Green Gram',
        'Q06_l': 'Black Gram',
        'Q06_m': 'Chickpea',
        'Q06_n': 'Other'
    }

    # Filter to only crops that exist in the data
    available_crops = {col: label for col, label in q06_labels.items() if col in df_demo.columns}

    if len(available_crops) > 0:
        # Create co-occurrence matrix
        crop_cols = list(available_crops.keys())
        crop_names = list(available_crops.values())

        # Initialize matrices
        import numpy as np
        total_respondents_fy26 = len(df_demo)
        total_respondents_fy25 = len(fy25_demo_filtered)
        cooccurrence_matrix_fy26 = np.zeros((len(crop_cols), len(crop_cols)))
        cooccurrence_matrix_fy25 = np.zeros((len(crop_cols), len(crop_cols)))

        for i, crop1 in enumerate(crop_cols):
            for j, crop2 in enumerate(crop_cols):
                if i == j:
                    # Diagonal: percentage of farmers growing this crop
                    count_fy26 = (df_demo[crop1] == 1.0).sum()
                    cooccurrence_matrix_fy26[i, j] = (count_fy26 / total_respondents_fy26 * 100) if total_respondents_fy26 > 0 else 0

                    if crop1 in fy25_demo_filtered.columns:
                        count_fy25 = (fy25_demo_filtered[crop1] == 1.0).sum()
                        cooccurrence_matrix_fy25[i, j] = (count_fy25 / total_respondents_fy25 * 100) if total_respondents_fy25 > 0 else 0
                elif i > j:
                    # Lower diagonal: percentage of farmers growing both crops
                    count_fy26 = ((df_demo[crop1] == 1.0) & (df_demo[crop2] == 1.0)).sum()
                    cooccurrence_matrix_fy26[i, j] = (count_fy26 / total_respondents_fy26 * 100) if total_respondents_fy26 > 0 else 0

                    if crop1 in fy25_demo_filtered.columns and crop2 in fy25_demo_filtered.columns:
                        count_fy25 = ((fy25_demo_filtered[crop1] == 1.0) & (fy25_demo_filtered[crop2] == 1.0)).sum()
                        cooccurrence_matrix_fy25[i, j] = (count_fy25 / total_respondents_fy25 * 100) if total_respondents_fy25 > 0 else 0
                else:
                    # Upper diagonal: set to NaN (will appear white)
                    cooccurrence_matrix_fy26[i, j] = np.nan
                    cooccurrence_matrix_fy25[i, j] = np.nan

        # Calculate difference matrix (FY26 - FY25)
        diff_matrix = cooccurrence_matrix_fy26 - cooccurrence_matrix_fy25

        # Create two columns for side-by-side heatmaps
        heatmap_col1, heatmap_col2 = st.columns(2)

        with heatmap_col1:
            st.markdown('#### FY26 Co-occurrence')
            # Create heatmap with RdBu colormap (Red to Blue)
            fig_crops = px.imshow(
                cooccurrence_matrix_fy26,
                labels=dict(x="Crop", y="Crop", color="Percentage (%)"),
                x=crop_names,
                y=crop_names,
                color_continuous_scale='RdBu_r',  # Red to Blue reversed
                aspect='auto',
                text_auto='.1f'
            )
            fig_crops.update_layout(
                xaxis_title='',
                yaxis_title='',
                height=500
            )
            fig_crops.update_xaxes(side='bottom', tickangle=-45)
            st.plotly_chart(fig_crops, use_container_width=True, key='crops_heatmap')

        with heatmap_col2:
            st.markdown('#### FY26 - FY25 Difference')
            # Create difference heatmap with PiYG colormap (Purple to Green)
            fig_diff = px.imshow(
                diff_matrix,
                labels=dict(x="Crop", y="Crop", color="Difference (pp)"),
                x=crop_names,
                y=crop_names,
                color_continuous_scale='PiYG',  # Purple-Yellow-Green (negative=purple, positive=green)
                aspect='auto',
                text_auto='.1f',
                color_continuous_midpoint=0
            )
            fig_diff.update_layout(
                xaxis_title='',
                yaxis_title='',
                height=500
            )
            fig_diff.update_xaxes(side='bottom', tickangle=-45)
            st.plotly_chart(fig_diff, use_container_width=True, key='crops_diff_heatmap')


with tab2:
    st.header('Digital Behavior & Engagement')

    # Create filters for Digital Behavior & Engagement
    tab2_filters = create_filters(df, 'tab2')

    # Filter data based on selections for Tab2
    df_tab2 = apply_filters(df, tab2_filters)

    # Display sample size at the top
    st.markdown(f"**Sample Size: n = {len(df_tab2)}**")
    st.markdown("---")

    # First row: Time of Day and Hours per day
    col1, col2 = st.columns(2)

    with col1:
        # Q11: Time of Day Usage Pattern
        st.subheader('Internet Usage by Time of Day')

        q11_labels = {
            'Q11_a': 'Morning',
            'Q11_b': 'Afternoon',
            'Q11_c': 'Evening',
            'Q11_d': 'Night'
        }

        time_order = ['Morning', 'Afternoon', 'Evening', 'Night']

        # Apply same filters to FY25 data
        fy25_tab2_filtered = apply_filters(df_fy25, tab2_filters)

        # FY26 data
        q11_data_fy26 = []
        for col_name, label in q11_labels.items():
            if col_name in df_tab2.columns:
                count = (df_tab2[col_name] == 1.0).sum()
                pct = (count / len(df_tab2) * 100) if len(df_tab2) > 0 else 0
                q11_data_fy26.append({
                    'Time of Day': label,
                    'Count': count,
                    'Percentage': pct
                })

        q11_df_fy26 = pd.DataFrame(q11_data_fy26)
        q11_df_fy26['Time of Day'] = pd.Categorical(q11_df_fy26['Time of Day'], categories=time_order, ordered=True)
        q11_df_fy26 = q11_df_fy26.sort_values('Time of Day')

        # FY25 data
        q11_data_fy25 = []
        for col_name, label in q11_labels.items():
            if col_name in fy25_tab2_filtered.columns:
                count = (fy25_tab2_filtered[col_name] == 1.0).sum()
                pct = (count / len(fy25_tab2_filtered) * 100) if len(fy25_tab2_filtered) > 0 else 0
                q11_data_fy25.append({
                    'Time of Day': label,
                    'Count': count,
                    'Percentage': pct
                })

        q11_df_fy25 = pd.DataFrame(q11_data_fy25)
        q11_df_fy25['Time of Day'] = pd.Categorical(q11_df_fy25['Time of Day'], categories=time_order, ordered=True)
        q11_df_fy25 = q11_df_fy25.sort_values('Time of Day')

        fig_q11 = go.Figure()

        # FY26 line (blue)
        fig_q11.add_trace(go.Scatter(
            x=q11_df_fy26['Time of Day'],
            y=q11_df_fy26['Percentage'],
            mode='lines+markers+text',
            name='FY26',
            line=dict(color='#4575b4', width=3),
            marker=dict(size=10, color='#4575b4', line=dict(color='#333', width=1)),
            text=q11_df_fy26['Percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='top center',
            textfont=dict(size=10),
            hovertemplate='FY26<br>%{x}<br>%{y:.1f}%<extra></extra>'
        ))

        # FY25 line (red)
        fig_q11.add_trace(go.Scatter(
            x=q11_df_fy25['Time of Day'],
            y=q11_df_fy25['Percentage'],
            mode='lines+markers+text',
            name='FY25',
            line=dict(color='#d73027', width=3, dash='dash'),
            marker=dict(size=10, color='#d73027', line=dict(color='#333', width=1)),
            text=q11_df_fy25['Percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='bottom center',
            textfont=dict(size=10),
            hovertemplate='FY25<br>%{x}<br>%{y:.1f}%<extra></extra>'
        ))

        max_pct = max(q11_df_fy26['Percentage'].max(), q11_df_fy25['Percentage'].max()) if len(q11_df_fy25) > 0 else q11_df_fy26['Percentage'].max()
        fig_q11.update_layout(
            xaxis_title='Time of Day',
            yaxis_title='Percentage of Users (%)',
            height=400,
            yaxis=dict(range=[0, max_pct * 1.2], showgrid=True, gridcolor='#e0e0e0'),
            xaxis=dict(categoryorder='array', categoryarray=time_order),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig_q11, use_container_width=True, key='time_of_day_line')

    with col2:
        # Q28: Hours per day - FY25 vs FY26 side by side
        st.subheader('Hours per day')
        col2a, col2b = st.columns(2)

        with col2a:
            st.markdown('**FY25**')
            if 'Q28' in fy25_tab2_filtered.columns:
                q28_counts_fy25 = fy25_tab2_filtered['Q28'].value_counts().reset_index()
                q28_counts_fy25.columns = ['Hours', 'Count']

                fig_q28_fy25 = px.pie(
                    q28_counts_fy25,
                    names='Hours',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_q28_fy25.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_q28_fy25.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_q28_fy25, use_container_width=True, key='hours_per_day_pie_fy25')

        with col2b:
            st.markdown('**FY26**')
            if 'Q28' in df_tab2.columns:
                q28_counts_fy26 = df_tab2['Q28'].value_counts().reset_index()
                q28_counts_fy26.columns = ['Hours', 'Count']

                fig_q28_fy26 = px.pie(
                    q28_counts_fy26,
                    names='Hours',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_q28_fy26.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_q28_fy26.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_q28_fy26, use_container_width=True, key='hours_per_day_pie_fy26')

    # Second row: Social Media Channels and Platform Usage by Hour
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        # Q10: Which social media channels are they using day to day? (UpSet plot with FY25 overlay)
        st.subheader('Which social media channels are they using day to day?')
        q10_labels = {
            'Q10_a': 'Facebook',
            'Q10_b': 'Youtube',
            'Q10_c': 'TikTok',
            'Q10_d': 'Viber',
            'Q10_e': 'Instagram',
            'Q10_f': 'Messenger',
            'Q10_g': 'Others'
        }

        # Prepare data for UpSet plot - FY26
        q10_cols = [col for col in q10_labels.keys() if col in df_tab2.columns]

        # Create a DataFrame for upset plot - need to convert to boolean
        upset_q10_data = df_tab2[q10_cols].copy()
        upset_q10_data = upset_q10_data.fillna(0)
        upset_q10_data = (upset_q10_data == 1.0)

        # Rename columns to friendly names
        rename_q10_map = {col: q10_labels[col] for col in q10_cols if col in q10_labels}
        upset_q10_data = upset_q10_data.rename(columns=rename_q10_map)

        # Count combinations for FY26
        from collections import Counter
        q10_combinations_fy26 = []
        for idx, row in upset_q10_data.iterrows():
            combo = tuple(sorted(col for col in upset_q10_data.columns if row[col]))
            if combo:
                q10_combinations_fy26.append(combo)

        q10_combo_counts_fy26 = Counter(q10_combinations_fy26)

        # Prepare FY25 data for the same combinations
        q10_cols_fy25 = [col for col in q10_labels.keys() if col in fy25_tab2_filtered.columns]
        upset_q10_data_fy25 = fy25_tab2_filtered[q10_cols_fy25].copy()
        upset_q10_data_fy25 = upset_q10_data_fy25.fillna(0)
        upset_q10_data_fy25 = (upset_q10_data_fy25 == 1.0)
        upset_q10_data_fy25 = upset_q10_data_fy25.rename(columns=rename_q10_map)

        q10_combinations_fy25 = []
        for idx, row in upset_q10_data_fy25.iterrows():
            combo = tuple(sorted(col for col in upset_q10_data_fy25.columns if row[col]))
            if combo:
                q10_combinations_fy25.append(combo)

        q10_combo_counts_fy25 = Counter(q10_combinations_fy25)

        # Get top 15 combinations from FY26
        q10_top_combos = sorted(q10_combo_counts_fy26.items(), key=lambda x: x[1], reverse=True)[:15]

        # Create upset-style visualization manually
        from plotly.subplots import make_subplots

        # Prepare data
        q10_combo_labels = []
        q10_combo_percentages_fy26 = []
        q10_combo_percentages_fy25 = []
        q10_matrix_data = {col: [] for col in upset_q10_data.columns}

        total_respondents_fy26 = len(df_tab2)
        total_respondents_fy25 = len(fy25_tab2_filtered)

        for combo, count in q10_top_combos:
            q10_combo_labels.append(' & '.join(combo) if len(combo) <= 2 else f"{len(combo)} channels")
            pct_fy26 = (count / total_respondents_fy26 * 100) if total_respondents_fy26 > 0 else 0
            q10_combo_percentages_fy26.append(pct_fy26)

            # Get FY25 count for the same combination
            count_fy25 = q10_combo_counts_fy25.get(combo, 0)
            pct_fy25 = (count_fy25 / total_respondents_fy25 * 100) if total_respondents_fy25 > 0 else 0
            q10_combo_percentages_fy25.append(pct_fy25)

            for col in upset_q10_data.columns:
                q10_matrix_data[col].append(1 if col in combo else 0)

        # Create figure with subplots
        fig_q10 = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.1,
            specs=[[{"type": "bar"}], [{"type": "scatter"}]]
        )

        # Top plot: bar chart of combination percentages (FY26)
        fig_q10.add_trace(
            go.Bar(
                x=list(range(len(q10_combo_percentages_fy26))),
                y=q10_combo_percentages_fy26,
                marker_color='#4575b4',
                name='FY26',
                width=0.6,
                text=[f'{p:.1f}%' for p in q10_combo_percentages_fy26],
                textposition='outside',
                hovertemplate='FY26<br>%{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Overlay FY25 line on top of bars
        fig_q10.add_trace(
            go.Scatter(
                x=list(range(len(q10_combo_percentages_fy25))),
                y=q10_combo_percentages_fy25,
                mode='lines+markers',
                name='FY25',
                line=dict(color='#d73027', width=2, dash='dash'),
                marker=dict(size=8, color='#d73027'),
                hovertemplate='FY25<br>%{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Bottom plot: matrix showing which channels are in each combination
        # Add connecting lines first
        for j in range(len(q10_matrix_data[list(upset_q10_data.columns)[0]])):
            active_indices = [i for i, col in enumerate(upset_q10_data.columns) if q10_matrix_data[col][j] == 1]

            if len(active_indices) > 1:
                fig_q10.add_trace(
                    go.Scatter(
                        x=[j] * len(active_indices),
                        y=active_indices,
                        mode='lines',
                        line=dict(color='#4575b4', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=2, col=1
                )

        # Add dots for active channels
        for i, col in enumerate(upset_q10_data.columns):
            x_vals = []
            y_vals = []

            for j in range(len(q10_matrix_data[col])):
                if q10_matrix_data[col][j] == 1:
                    x_vals.append(j)
                    y_vals.append(i)

            fig_q10.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    marker=dict(size=10, color='#4575b4', line=dict(color='#ffffff', width=1)),
                    showlegend=False,
                    name=col,
                    hovertemplate=f'{col}<extra></extra>'
                ),
                row=2, col=1
            )

        # Update layout
        fig_q10.update_xaxes(
            showticklabels=False,
            showgrid=False,
            range=[-0.5, len(q10_combo_percentages_fy26) - 0.5],
            row=1, col=1
        )
        fig_q10.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, len(q10_combo_percentages_fy26) - 0.5],
            row=2, col=1
        )
        fig_q10.update_yaxes(
            title_text="Percentage (%)",
            showgrid=True,
            gridcolor='#e0e0e0',
            row=1, col=1
        )
        fig_q10.update_yaxes(
            ticktext=list(upset_q10_data.columns),
            tickvals=list(range(len(upset_q10_data.columns))),
            showgrid=False,
            zeroline=False,
            row=2, col=1
        )

        fig_q10.update_layout(
            height=500,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_q10, use_container_width=True, key='social_media_upset')

    with col4:
        # Platform Usage by Hour - paired stacked bar chart (FY25 vs FY26)
        st.subheader('Platform Usage by Hour')

        # Calculate platform usage stats for both years
        def calc_platform_time_stats(data, platform_cols):
            stats = []
            for col, platform in platform_cols.items():
                if col in data.columns and 'Q28' in data.columns:
                    users = data[data[col] == 1.0]
                    time_dist = users['Q28'].value_counts()

                    lt2h = time_dist.get('Less than 2 hours', 0)
                    t2_4h = time_dist.get('2-4 hours', 0)
                    gt4h = time_dist.get('more than 4 hours', 0)

                    total = lt2h + t2_4h + gt4h
                    if total > 0:
                        pct_lt2h = lt2h / total * 100
                        pct_2_4h = t2_4h / total * 100
                        pct_gt4h = gt4h / total * 100
                    else:
                        pct_lt2h = pct_2_4h = pct_gt4h = 0

                    stats.append({
                        'Platform': platform,
                        'Pct_LessThan2h': pct_lt2h,
                        'Pct_2to4h': pct_2_4h,
                        'Pct_MoreThan4h': pct_gt4h
                    })
            return pd.DataFrame(stats)

        platform_cols = {
            'Q10_a': 'Facebook',
            'Q10_b': 'Youtube',
            'Q10_c': 'TikTok',
            'Q10_d': 'Viber',
            'Q10_e': 'Instagram',
            'Q10_f': 'Messenger',
            'Q10_g': 'Others'
        }

        # Calculate for FY26 and FY25
        fy26_stats = calc_platform_time_stats(df_tab2, platform_cols)
        fy25_stats = calc_platform_time_stats(fy25_tab2_filtered, platform_cols)

        if len(fy26_stats) > 0 and len(fy25_stats) > 0:
            # Create x-axis labels with paired bars
            platforms = fy26_stats['Platform'].tolist()
            x_positions = []
            x_labels = []
            bar_width = 0.35

            for i, platform in enumerate(platforms):
                x_positions.append(i * 2)  # FY26 position
                x_positions.append(i * 2 + 0.8)  # FY25 position
                x_labels.extend([f'{platform}', ''])

            fig_platform_time = go.Figure()

            # FY25 bars (left of each pair) - lighter colors
            fy25_x = [i * 2 for i in range(len(platforms))]
            fig_platform_time.add_trace(go.Bar(
                name='<2h (FY25)',
                x=fy25_x,
                y=fy25_stats['Pct_LessThan2h'],
                marker_color='#a8e6cf',  # Lighter green
                width=bar_width,
                hovertemplate='FY25<br>%{customdata}<br><2 hours: %{y:.1f}%<extra></extra>',
                customdata=platforms,
                legendgroup='lt2h'
            ))
            fig_platform_time.add_trace(go.Bar(
                name='2-4h (FY25)',
                x=fy25_x,
                y=fy25_stats['Pct_2to4h'],
                marker_color='#fad390',  # Lighter orange
                width=bar_width,
                hovertemplate='FY25<br>%{customdata}<br>2-4 hours: %{y:.1f}%<extra></extra>',
                customdata=platforms,
                legendgroup='2_4h'
            ))
            fig_platform_time.add_trace(go.Bar(
                name='>4h (FY25)',
                x=fy25_x,
                y=fy25_stats['Pct_MoreThan4h'],
                marker_color='#f8b4b4',  # Lighter red
                width=bar_width,
                hovertemplate='FY25<br>%{customdata}<br>>4 hours: %{y:.1f}%<extra></extra>',
                customdata=platforms,
                legendgroup='gt4h'
            ))

            # FY26 bars (right of each pair)
            fy26_x = [i * 2 + 0.4 for i in range(len(platforms))]
            fig_platform_time.add_trace(go.Bar(
                name='<2h (FY26)',
                x=fy26_x,
                y=fy26_stats['Pct_LessThan2h'],
                marker_color='#2ecc71',
                width=bar_width,
                hovertemplate='FY26<br>%{customdata}<br><2 hours: %{y:.1f}%<extra></extra>',
                customdata=platforms,
                legendgroup='lt2h'
            ))
            fig_platform_time.add_trace(go.Bar(
                name='2-4h (FY26)',
                x=fy26_x,
                y=fy26_stats['Pct_2to4h'],
                marker_color='#f39c12',
                width=bar_width,
                hovertemplate='FY26<br>%{customdata}<br>2-4 hours: %{y:.1f}%<extra></extra>',
                customdata=platforms,
                legendgroup='2_4h'
            ))
            fig_platform_time.add_trace(go.Bar(
                name='>4h (FY26)',
                x=fy26_x,
                y=fy26_stats['Pct_MoreThan4h'],
                marker_color='#e74c3c',
                width=bar_width,
                hovertemplate='FY26<br>%{customdata}<br>>4 hours: %{y:.1f}%<extra></extra>',
                customdata=platforms,
                legendgroup='gt4h'
            ))

            fig_platform_time.update_layout(
                barmode='stack',
                xaxis_title='',
                yaxis_title='% of Platform Users',
                height=500,
                xaxis=dict(
                    tickmode='array',
                    tickvals=[i * 2 + 0.2 for i in range(len(platforms))],
                    ticktext=platforms,
                    tickangle=-45
                ),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                margin=dict(l=10, r=10, t=60, b=80)
            )

            # Add FY25/FY26 annotations below bars
            for i, platform in enumerate(platforms):
                fig_platform_time.add_annotation(
                    x=i * 2, y=-8, text='25', showarrow=False, font=dict(size=8), yref='y'
                )
                fig_platform_time.add_annotation(
                    x=i * 2 + 0.4, y=-8, text='26', showarrow=False, font=dict(size=8), yref='y'
                )

            st.plotly_chart(fig_platform_time, use_container_width=True, key='platform_time_distribution')
        else:
            st.warning('Platform usage data not available.')

    # Third row: Phone sharing and Poor connection
    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        # Q07: Who do they share their phone with? - FY25 vs FY26 side by side
        st.subheader('Who do they share their phone with?')
        q07_labels = {
            'Q07_a': 'Children',
            'Q07_b': 'Spouse',
            'Q07_c': 'Cousin',
            'Q07_d': 'Parents',
            'Q07_e': 'Others',
            'Q07_f': "I don't share with anyone"
        }

        col5a, col5b = st.columns(2)

        with col5a:
            st.markdown('**FY25**')
            q07_data_fy25 = []
            for col_name, label in q07_labels.items():
                if col_name in fy25_tab2_filtered.columns:
                    count = (fy25_tab2_filtered[col_name] == 1.0).sum()
                    q07_data_fy25.append({'Category': label, 'Count': count})

            q07_df_fy25 = pd.DataFrame(q07_data_fy25)

            fig_q07_fy25 = px.pie(
                q07_df_fy25,
                names='Category',
                values='Count',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_q07_fy25.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
            fig_q07_fy25.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_q07_fy25, use_container_width=True, key='phone_sharing_pie_fy25')

        with col5b:
            st.markdown('**FY26**')
            q07_data_fy26 = []
            for col_name, label in q07_labels.items():
                if col_name in df_tab2.columns:
                    count = (df_tab2[col_name] == 1.0).sum()
                    q07_data_fy26.append({'Category': label, 'Count': count})

            q07_df_fy26 = pd.DataFrame(q07_data_fy26)

            fig_q07_fy26 = px.pie(
                q07_df_fy26,
                names='Category',
                values='Count',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_q07_fy26.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
            fig_q07_fy26.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_q07_fy26, use_container_width=True, key='phone_sharing_pie_fy26')

    with col6:
        # Q27: Level of poor connection - FY25 vs FY26 side by side
        st.subheader('Level of poor connection')

        # Split long labels intelligently at word boundaries
        def split_label(text, max_len=15):
            if len(text) <= max_len:
                return text

            words = text.split(' ')
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + (1 if current_line else 0) > max_len:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        lines.append(word)
                        current_length = 0
                else:
                    current_line.append(word)
                    current_length += len(word) + (1 if len(current_line) > 1 else 0)

            if current_line:
                lines.append(' '.join(current_line))

            return '<br>'.join(lines)

        col6a, col6b = st.columns(2)

        with col6a:
            st.markdown('**FY25**')
            if 'Q27' in fy25_tab2_filtered.columns:
                q27_counts_fy25 = fy25_tab2_filtered['Q27'].value_counts().reset_index()
                q27_counts_fy25.columns = ['Connection Level', 'Count']
                q27_counts_fy25['Connection Level'] = q27_counts_fy25['Connection Level'].str.capitalize()
                q27_counts_fy25['Connection Level'] = q27_counts_fy25['Connection Level'].apply(split_label)

                fig_q27_fy25 = px.pie(
                    q27_counts_fy25,
                    names='Connection Level',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_q27_fy25.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_q27_fy25.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_q27_fy25, use_container_width=True, key='connection_level_pie_fy25')

        with col6b:
            st.markdown('**FY26**')
            if 'Q27' in df_tab2.columns:
                q27_counts_fy26 = df_tab2['Q27'].value_counts().reset_index()
                q27_counts_fy26.columns = ['Connection Level', 'Count']
                q27_counts_fy26['Connection Level'] = q27_counts_fy26['Connection Level'].str.capitalize()
                q27_counts_fy26['Connection Level'] = q27_counts_fy26['Connection Level'].apply(split_label)

                fig_q27_fy26 = px.pie(
                    q27_counts_fy26,
                    names='Connection Level',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_q27_fy26.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_q27_fy26.update_layout(height=350, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_q27_fy26, use_container_width=True, key='connection_level_pie_fy26')


with tab3:
    st.header('Internet Usage Behaviors')

    # Create filters for Internet Usage Behaviors
    tab3_filters = create_filters(df, 'tab3')

    # Filter data based on selections for Tab3
    df_tab3 = apply_filters(df, tab3_filters)

    # Filter FY25 data for Tab3
    df_fy25_tab3 = apply_filters(df_fy25, tab3_filters)

    # Display sample size as subtle caption
    st.caption(f"n = {len(df_tab3):,} (FY26) · {len(df_fy25_tab3):,} (FY25)")
    st.markdown("---")

    # Create sub-tabs for each platform
    platform_tabs = st.tabs(['Facebook', 'TikTok', 'Viber', 'YouTube', 'Other'])

    # Facebook Tab
    with platform_tabs[0]:
        # Define confidence level mapping for Facebook
        fb_confidence_map = {
            'using news feed only': 'Newsfeed only',
            'Can use search': 'Can use search',
            'Can use Facebook actively (post, share, comment)': 'Engage Actively'
        }

        create_platform_analysis(
            df=df_tab3,
            df_fy25=df_fy25_tab3,
            platform_name='Facebook',
            platform_key='fb',
            platform_col='Q10_a',
            usecase_col='Q12',
            confidence_col='Q14',
            challenge_col='Q13',
            confidence_map=fb_confidence_map
        )

    # TikTok Tab
    with platform_tabs[1]:
        create_platform_analysis(
            df=df_tab3,
            df_fy25=df_fy25_tab3,
            platform_name='TikTok',
            platform_key='tt',
            platform_col='Q10_c',
            usecase_col='Q18',
            confidence_col='Q20',
            challenge_col='Q19'
        )

    # Viber Tab
    with platform_tabs[2]:
        create_platform_analysis(
            df=df_tab3,
            df_fy25=df_fy25_tab3,
            platform_name='Viber',
            platform_key='vb',
            platform_col='Q10_d',
            usecase_col='Q21',
            confidence_col='Q23',
            challenge_col='Q22'
        )

    # YouTube Tab
    with platform_tabs[3]:
        create_platform_analysis(
            df=df_tab3,
            df_fy25=df_fy25_tab3,
            platform_name='YouTube',
            platform_key='yt',
            platform_col='Q10_b',
            usecase_col='Q15',
            confidence_col='Q17',
            challenge_col='Q16'
        )

    # Other Social Media Tab
    with platform_tabs[4]:
        create_platform_analysis(
            df=df_tab3,
            df_fy25=df_fy25_tab3,
            platform_name='Other Social Media',
            platform_key='other',
            platform_col='Q10_g',
            usecase_col='Q24',
            confidence_col='Q26',
            challenge_col='Q25'
        )

with tab4:
    st.header('Brand & Product Awareness')

    # Create filters for Brand & Product Awareness (applies to all subsections)
    tab4_filters = create_filters(df, 'tab4')

    # Filter data based on selections
    brand_filtered = apply_filters(df, tab4_filters)

    # Display sample size at the top
    st.markdown(f"**Sample Size: n = {len(brand_filtered)}**")
    st.markdown("---")

    # Brand Awareness Section
    st.subheader('Brand Awareness')

    # Combined Agricultural Brands: Heard vs Used
    st.markdown('#### Agricultural Brands: Awareness vs Usage')

    # Q36 and Q37: Agricultural brands heard of vs used
    q36_brand_labels = {
        'Q36_f': 'TRI-S',
        'Q36_g': 'Tricho Power',
        'Q36_h': 'Transform',
        'Q36_i': 'Sote Phwar',
        'Q36_j': 'Shwe Tet Lan',
        'Q36_k': 'Hatake',
        'Q36_l': 'Ywat Lwint',
        'Q36_m': 'Pyan Lwar',
        'Q36_n': 'Doh Kyay Let',
        'Q36_o': 'Eco-Vital Bio-stimulant',
        'Q36_p': 'Other'
    }

    q37_brand_labels = {
        'Q37_c': 'TRI-S',
        'Q37_d': 'Tricho Power',
        'Q37_e': 'Transform',
        'Q37_f': 'Sote Phwar',
        'Q37_g': 'Shwe Tet Lan',
        'Q37_h': 'Hatake',
        'Q37_i': 'Ywat Lwint',
        'Q37_j': 'Pyan Lwar',
        'Q37_k': 'Doh Kyay Let',
        'Q37_l': 'Eco-Vital Bio-stimulant',
        'Q37_m': 'Other'
    }

    # Combine data for both heard and used
    combined_brands_data = []

    for col_name, label in q36_brand_labels.items():
        if col_name in brand_filtered.columns:
            count = (brand_filtered[col_name] == 1.0).sum()
            pct = (count / len(brand_filtered) * 100) if len(brand_filtered) > 0 else 0
            combined_brands_data.append({'Brand': label, 'Percentage': pct, 'Type': 'Heard Of'})

    for col_name, label in q37_brand_labels.items():
        if col_name in brand_filtered.columns:
            count = (brand_filtered[col_name] == 1.0).sum()
            pct = (count / len(brand_filtered) * 100) if len(brand_filtered) > 0 else 0
            combined_brands_data.append({'Brand': label, 'Percentage': pct, 'Type': 'Used'})

    combined_brands_df = pd.DataFrame(combined_brands_data)

    # Sort brands by average percentage (descending for vertical bars)
    avg_brand_pct = combined_brands_df.groupby('Brand')['Percentage'].mean().reset_index()
    avg_brand_pct.columns = ['Brand', 'Avg_Pct']
    avg_brand_pct = avg_brand_pct.sort_values('Avg_Pct', ascending=False)

    combined_brands_df['Brand'] = pd.Categorical(combined_brands_df['Brand'], categories=avg_brand_pct['Brand'].tolist(), ordered=True)
    combined_brands_df = combined_brands_df.sort_values(['Brand', 'Type'])

    fig_combined_brands = px.bar(
        combined_brands_df,
        x='Brand',
        y='Percentage',
        color='Type',
        barmode='group',
        color_discrete_map={'Heard Of': '#0f4c3a', 'Used': '#5a8f7b'},
        text='Percentage',
        category_orders={'Type': ['Heard Of', 'Used']}
    )
    fig_combined_brands.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_combined_brands.update_layout(
        xaxis_title='',
        yaxis_title='Percentage (%)',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis={'tickangle': -45}
    )
    st.plotly_chart(fig_combined_brands, use_container_width=True, key='combined_brands_chart')

    st.markdown('---')

    # Product Awareness Section
    st.subheader('Product Awareness')

    # Use the same filtered data from tab4_filters
    prod_filtered = brand_filtered

    # Combined Yetagon Products: Heard vs Used
    st.markdown('#### Yetagon Products/Services: Awareness vs Usage')

    # Q31 and Q33: Yetagon products heard of vs used
    q31_labels = {
        'Q31_a': 'Yetagon Irrigation',
        'Q31_b': 'Yetagon EM/Zarmani',
        'Q31_c': 'Yetagon Trichoderma/Barhmati',
        'Q31_d': 'Yetagon Tele-agronomy',
        'Q31_e': 'Yetagon Sun-kissed',
        'Q31_f': 'Yetagon Fish Amino',
        'Q31_g': 'Po Chat',
        'Q31_h': 'Messenger',
        'Q31_i': 'Digital farm practices',
        'Q31_j': 'None of the above',
        'Q31_k': 'EM',
        'Q31_l': 'Other'
    }

    q33_labels = {
        'Q33_a': 'Yetagon Irrigation',
        'Q33_b': 'Yetagon EM/Zarmani',
        'Q33_c': 'Yetagon Trichoderma/Barhmati',
        'Q33_d': 'Yetagon Tele-agronomy',
        'Q33_h': 'Yetagon Sun-kissed',
        'Q33_i': 'Yetagon Fish Amino',
        'Q33_j': 'Po Chat',
        'Q33_k': 'Messenger',
        'Q33_e': 'Digital farm practices',
        'Q33_f': 'None of the above',
        'Q33_g': 'Other'
    }

    # Combine data for both heard and used
    combined_products_data = []

    for col_name, label in q31_labels.items():
        if col_name in prod_filtered.columns:
            count = (prod_filtered[col_name] == 1.0).sum()
            pct = (count / len(prod_filtered) * 100) if len(prod_filtered) > 0 else 0
            combined_products_data.append({'Product': label, 'Percentage': pct, 'Type': 'Heard Of'})

    for col_name, label in q33_labels.items():
        if col_name in prod_filtered.columns:
            count = (prod_filtered[col_name] == 1.0).sum()
            pct = (count / len(prod_filtered) * 100) if len(prod_filtered) > 0 else 0
            combined_products_data.append({'Product': label, 'Percentage': pct, 'Type': 'Used'})

    combined_products_df = pd.DataFrame(combined_products_data)

    # Sort products by average percentage (descending for vertical bars)
    avg_pct = combined_products_df.groupby('Product')['Percentage'].mean().reset_index()
    avg_pct.columns = ['Product', 'Avg_Pct']
    avg_pct = avg_pct.sort_values('Avg_Pct', ascending=False)

    combined_products_df['Product'] = pd.Categorical(combined_products_df['Product'], categories=avg_pct['Product'].tolist(), ordered=True)
    combined_products_df = combined_products_df.sort_values(['Product', 'Type'])

    fig_combined_products = px.bar(
        combined_products_df,
        x='Product',
        y='Percentage',
        color='Type',
        barmode='group',
        color_discrete_map={'Heard Of': '#0f4c3a', 'Used': '#5a8f7b'},
        text='Percentage',
        category_orders={'Type': ['Heard Of', 'Used']}
    )
    fig_combined_products.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_combined_products.update_layout(
        xaxis_title='',
        yaxis_title='Percentage (%)',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis={'tickangle': -45}
    )
    st.plotly_chart(fig_combined_products, use_container_width=True, key='combined_products_chart')

    # Combined Techniques: Heard vs Used
    st.markdown('#### Farming Techniques: Awareness vs Usage')

    # Q32 and Q34: Techniques heard of vs used
    q32_labels = {
        'Q32_a': 'No Burn Rice Farming',
        'Q32_b': 'Salt Water Seed Selection',
        'Q32_c': 'Basal Fertilizer Usage for Rice',
        'Q32_d': 'Mid-season Fertilizer Usage for Rice',
        'Q32_e': 'Paddy Liming Acid',
        'Q32_f': 'Gypsum Application',
        'Q32_g': 'Boron Foliar Spray',
        'Q32_h': 'Epsom Salt Foliar Spray',
        'Q32_i': 'Neem Pesticide',
        'Q32_j': 'Fish Amino',
        'Q32_k': 'Other'
    }

    q34_labels = {
        'Q34_a': 'No Burn Rice Farming',
        'Q34_b': 'Salt Water Seed Selection',
        'Q34_c': 'Basal Fertilizer Usage for Rice',
        'Q34_d': 'Mid-season Fertilizer Usage for Rice',
        'Q34_e': 'Paddy Liming Acid',
        'Q34_f': 'Gypsum Application',
        'Q34_g': 'Boron Foliar Spray',
        'Q34_h': 'Epsom Salt Foliar Spray',
        'Q34_i': 'Neem Pesticide',
        'Q34_j': 'Fish Amino',
        'Q34_k': 'Other'
    }

    # Combine data for both heard and used
    combined_techniques_data = []

    for col_name, label in q32_labels.items():
        if col_name in prod_filtered.columns:
            count = (prod_filtered[col_name] == 1.0).sum()
            pct = (count / len(prod_filtered) * 100) if len(prod_filtered) > 0 else 0
            combined_techniques_data.append({'Technique': label, 'Percentage': pct, 'Type': 'Heard Of'})

    for col_name, label in q34_labels.items():
        if col_name in prod_filtered.columns:
            count = (prod_filtered[col_name] == 1.0).sum()
            pct = (count / len(prod_filtered) * 100) if len(prod_filtered) > 0 else 0
            combined_techniques_data.append({'Technique': label, 'Percentage': pct, 'Type': 'Used'})

    combined_techniques_df = pd.DataFrame(combined_techniques_data)

    # Sort techniques by average percentage (descending for vertical bars)
    avg_tech_pct = combined_techniques_df.groupby('Technique')['Percentage'].mean().reset_index()
    avg_tech_pct.columns = ['Technique', 'Avg_Pct']
    avg_tech_pct = avg_tech_pct.sort_values('Avg_Pct', ascending=False)

    combined_techniques_df['Technique'] = pd.Categorical(combined_techniques_df['Technique'], categories=avg_tech_pct['Technique'].tolist(), ordered=True)
    combined_techniques_df = combined_techniques_df.sort_values(['Technique', 'Type'])

    fig_combined_techniques = px.bar(
        combined_techniques_df,
        x='Technique',
        y='Percentage',
        color='Type',
        barmode='group',
        color_discrete_map={'Heard Of': '#0f4c3a', 'Used': '#5a8f7b'},
        text='Percentage',
        category_orders={'Type': ['Heard Of', 'Used']}
    )
    fig_combined_techniques.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_combined_techniques.update_layout(
        xaxis_title='',
        yaxis_title='Percentage (%)',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis={'tickangle': -45}
    )
    st.plotly_chart(fig_combined_techniques, use_container_width=True, key='combined_techniques_chart')

    st.markdown('---')

    # Digital Products Analysis Section
    st.subheader('Digital Products Analysis')
    st.caption('Note: Any technique (Q32/Q34), Po Chat (Q31_g), Messenger (Q31_h), Digital farm practices (Q31_i); FY25 does not have a "Used" category for Po Chat (Q33_j does not exist in FY25 survey) or Messenger (Q33_k does not exist in FY25 survey)')

    # Define digital product columns
    # Heard of: Any Q32 technique OR Q31_g OR Q31_h OR Q31_i
    q32_heard_cols = [f'Q32_{chr(97+i)}' for i in range(11)]  # Q32_a through Q32_k
    digital_heard_cols = q32_heard_cols + ['Q31_g', 'Q31_h', 'Q31_i']

    # Used: Any Q34 technique OR Q33_j (Po Chat used) OR Q33_k (Messenger used) OR Q33_e (Digital farm practices used)
    q34_used_cols = [f'Q34_{chr(97+i)}' for i in range(11)]  # Q34_a through Q34_k
    digital_used_cols = q34_used_cols + ['Q33_j', 'Q33_k', 'Q33_e']

    # Create two columns for side-by-side display
    digital_col1, digital_col2 = st.columns(2)

    with digital_col1:
        st.markdown('#### Current Analysis')

        # Calculate number of digital products heard of per person
        heard_digital = prod_filtered.copy()
        heard_digital['Heard_Count'] = 0
        for col in digital_heard_cols:
            if col in heard_digital.columns:
                heard_digital['Heard_Count'] = heard_digital['Heard_Count'] + heard_digital[col].fillna(0).astype(int)

        # Calculate number of digital products used per person
        used_digital = prod_filtered.copy()
        used_digital['Used_Count'] = 0
        for col in digital_used_cols:
            if col in used_digital.columns:
                used_digital['Used_Count'] = used_digital['Used_Count'] + used_digital[col].fillna(0).astype(int)

        # Count distribution for Heard Of
        heard_dist = heard_digital['Heard_Count'].value_counts().sort_index()
        heard_1 = heard_dist.get(1, 0)
        heard_2 = heard_dist.get(2, 0)
        heard_3 = heard_dist.get(3, 0)
        heard_4plus = sum(heard_dist.get(i, 0) for i in range(4, 20))  # 4 or more

        # Count distribution for Used
        used_dist = used_digital['Used_Count'].value_counts().sort_index()
        used_1 = used_dist.get(1, 0)
        used_2 = used_dist.get(2, 0)
        used_3 = used_dist.get(3, 0)
        used_4plus = sum(used_dist.get(i, 0) for i in range(4, 20))  # 4 or more

        total = len(prod_filtered)

        # Create stacked bar chart data
        digital_products_data = pd.DataFrame({
            'Category': ['Heard Of', 'Used'],
            '1 Product': [heard_1/total*100 if total > 0 else 0, used_1/total*100 if total > 0 else 0],
            '2 Products': [heard_2/total*100 if total > 0 else 0, used_2/total*100 if total > 0 else 0],
            '3 Products': [heard_3/total*100 if total > 0 else 0, used_3/total*100 if total > 0 else 0],
            '4+ Products': [heard_4plus/total*100 if total > 0 else 0, used_4plus/total*100 if total > 0 else 0]
        })

        # Create stacked bar chart
        fig_digital_products = go.Figure()

        colors = ['#e8f4f1', '#5a8f7b', '#0f4c3a', '#073d2a']  # Light to dark green

        for i, product_count in enumerate(['1 Product', '2 Products', '3 Products', '4+ Products']):
            fig_digital_products.add_trace(go.Bar(
                name=product_count,
                x=digital_products_data['Category'],
                y=digital_products_data[product_count],
                marker_color=colors[i],
                text=[f'{val:.1f}%' if val > 0 else '' for val in digital_products_data[product_count]],
                textposition='inside',
                hovertemplate='%{x}<br>' + product_count + ': %{y:.1f}%<extra></extra>'
            ))

        fig_digital_products.update_layout(
            barmode='stack',
            xaxis_title='',
            yaxis_title='Percentage (%)',
            height=500,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        st.plotly_chart(fig_digital_products, use_container_width=True, key='digital_products_chart')

    with digital_col2:
        st.markdown('#### FY25 vs FY26 Comparison')

        # Apply the same filters to FY25 data if not already done
        if 'fy25_filtered' not in locals():
            fy25_filtered = apply_filters(df_fy25, tab4_filters)

        # Calculate Digital Products for FY26
        # Heard Of
        fy26_digital_heard = prod_filtered.copy()
        fy26_digital_heard['Digital_Heard'] = False
        for col in digital_heard_cols:
            if col in fy26_digital_heard.columns:
                fy26_digital_heard['Digital_Heard'] = fy26_digital_heard['Digital_Heard'] | (fy26_digital_heard[col] == 1.0)
        fy26_heard_pct = (fy26_digital_heard['Digital_Heard'].sum() / len(fy26_digital_heard) * 100) if len(fy26_digital_heard) > 0 else 0

        # Used
        fy26_digital_used = prod_filtered.copy()
        fy26_digital_used['Digital_Used'] = False
        for col in digital_used_cols:
            if col in fy26_digital_used.columns:
                fy26_digital_used['Digital_Used'] = fy26_digital_used['Digital_Used'] | (fy26_digital_used[col] == 1.0)
        fy26_used_pct = (fy26_digital_used['Digital_Used'].sum() / len(fy26_digital_used) * 100) if len(fy26_digital_used) > 0 else 0

        # Calculate Digital Products for FY25
        # Heard Of
        fy25_digital_heard = fy25_filtered.copy()
        fy25_digital_heard['Digital_Heard'] = False
        for col in digital_heard_cols:
            if col in fy25_digital_heard.columns:
                fy25_digital_heard['Digital_Heard'] = fy25_digital_heard['Digital_Heard'] | (fy25_digital_heard[col] == 1.0)
        fy25_heard_pct = (fy25_digital_heard['Digital_Heard'].sum() / len(fy25_digital_heard) * 100) if len(fy25_digital_heard) > 0 else 0

        # Used
        fy25_digital_used = fy25_filtered.copy()
        fy25_digital_used['Digital_Used'] = False
        for col in digital_used_cols:
            if col in fy25_digital_used.columns:
                fy25_digital_used['Digital_Used'] = fy25_digital_used['Digital_Used'] | (fy25_digital_used[col] == 1.0)
        fy25_used_pct = (fy25_digital_used['Digital_Used'].sum() / len(fy25_digital_used) * 100) if len(fy25_digital_used) > 0 else 0

        # Create slope chart for Digital Products
        fig_digital_yoy = go.Figure()

        # FY26 line (Heard Of to Used)
        fig_digital_yoy.add_trace(go.Scatter(
            x=[0, 1],
            y=[fy26_heard_pct, fy26_used_pct],
            mode='lines+markers',
            name='FY26',
            line=dict(color='#0f4c3a', width=3),
            marker=dict(size=12),
            hovertemplate='FY26<br>%{y:.1f}%<extra></extra>'
        ))

        # FY25 line (Heard Of to Used)
        fig_digital_yoy.add_trace(go.Scatter(
            x=[0, 1],
            y=[fy25_heard_pct, fy25_used_pct],
            mode='lines+markers',
            name='FY25',
            line=dict(color='#8fc1e3', width=3, dash='dash'),
            marker=dict(size=12),
            hovertemplate='FY25<br>%{y:.1f}%<extra></extra>'
        ))

        fig_digital_yoy.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Heard Of', 'Used'],
                showgrid=True
            ),
            yaxis=dict(
                title='Percentage (%)',
                showgrid=True,
                gridcolor='#e0e0e0'
            ),
            height=500,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )

        st.plotly_chart(fig_digital_yoy, use_container_width=True, key='digital_products_yoy')

    st.markdown('---')

    # FY25 vs FY26 Comparison Section
    st.subheader('Year-over-Year Comparison: FY25 vs FY26')

    # Apply the same filters to FY25 data
    fy25_filtered = apply_filters(df_fy25, tab4_filters)

    # Helper function to calculate percentages for a given year
    def calculate_year_percentages(df_year, question_cols, label_map):
        """Calculate percentages for heard/used questions across products"""
        results = []
        total = len(df_year)

        for col_name, label in label_map.items():
            if col_name in df_year.columns:
                count = (df_year[col_name] == 1.0).sum()
                pct = (count / total * 100) if total > 0 else 0
                results.append({'Product': label, 'Percentage': pct})

        return pd.DataFrame(results)

    # Create slope chart for Yetagon Products
    st.markdown(f'#### Yetagon Products/Services: FY25 vs FY26')

    yetagon_product_labels = {
        'Q31_a': 'Yetagon Irrigation',
        'Q31_b': 'Yetagon EM/Zarmani',
        'Q31_c': 'Yetagon Trichoderma',
        'Q31_d': 'Yetagon Tele-agronomy',
        'Q31_i': 'Digital farm practices',
        'Q31_g': 'Po Chat'
    }

    yetagon_used_labels = {
        'Q33_a': 'Yetagon Irrigation',
        'Q33_b': 'Yetagon EM/Zarmani',
        'Q33_c': 'Yetagon Trichoderma',
        'Q33_d': 'Yetagon Tele-agronomy',
        'Q33_e': 'Digital farm practices',
        'Q33_j': 'Po Chat'
    }

    # Calculate percentages for FY25 and FY26 using filtered data
    fy26_heard = calculate_year_percentages(prod_filtered, yetagon_product_labels, yetagon_product_labels)
    fy25_heard = calculate_year_percentages(fy25_filtered, yetagon_product_labels, yetagon_product_labels)

    fy26_used = calculate_year_percentages(prod_filtered, yetagon_used_labels, yetagon_used_labels)
    fy25_used = calculate_year_percentages(fy25_filtered, yetagon_used_labels, yetagon_used_labels)

    # Create slope chart
    fig_yetagon_comparison = go.Figure()

    # Get unique products (matching between heard and used)
    products = list(yetagon_product_labels.values())
    n_products = len(products)

    # Create x-positions: each product has 2 positions (Heard Of, Used)
    # Product 0: x=0 (Heard), x=1 (Used)
    # Product 1: x=3 (Heard), x=4 (Used)
    # Product 2: x=6 (Heard), x=7 (Used), etc.
    x_spacing = 3  # spacing between products

    all_tick_positions = []
    all_tick_labels = []
    product_positions = []  # Center position for each product label

    # Plot for each product
    for i, product in enumerate(products):
        # Calculate x positions for this product
        x_heard = i * x_spacing
        x_used = i * x_spacing + 1

        all_tick_positions.extend([x_heard, x_used])
        all_tick_labels.extend(['Heard Of', 'Used'])
        product_positions.append(i * x_spacing + 0.5)  # Center between heard and used

        # Get FY26 data
        fy26_h = fy26_heard[fy26_heard['Product'] == product]['Percentage'].values
        fy26_u = fy26_used[fy26_used['Product'] == product]['Percentage'].values

        # Get FY25 data
        fy25_h = fy25_heard[fy25_heard['Product'] == product]['Percentage'].values
        fy25_u = fy25_used[fy25_used['Product'] == product]['Percentage'].values

        # FY26 line (Heard Of to Used)
        if len(fy26_h) > 0 and len(fy26_u) > 0:
            fig_yetagon_comparison.add_trace(go.Scatter(
                x=[x_heard, x_used],
                y=[fy26_h[0], fy26_u[0]],
                mode='lines+markers',
                name=f'FY26' if i == 0 else '',
                line=dict(color='#0f4c3a', width=2),
                marker=dict(size=8),
                showlegend=(i == 0),
                legendgroup='FY26',
                hovertemplate=f'{product}<br>%{{y:.1f}}%<extra></extra>'
            ))
        elif len(fy26_h) > 0 or len(fy26_u) > 0:
            # Plot individual points if only one exists
            x_points = []
            y_points = []
            if len(fy26_h) > 0:
                x_points.append(x_heard)
                y_points.append(fy26_h[0])
            if len(fy26_u) > 0:
                x_points.append(x_used)
                y_points.append(fy26_u[0])
            fig_yetagon_comparison.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='markers',
                name=f'FY26' if i == 0 else '',
                marker=dict(color='#0f4c3a', size=8),
                showlegend=(i == 0),
                legendgroup='FY26',
                hovertemplate=f'{product}<br>%{{y:.1f}}%<extra></extra>'
            ))

        # FY25 line (Heard Of to Used)
        if len(fy25_h) > 0 and len(fy25_u) > 0:
            fig_yetagon_comparison.add_trace(go.Scatter(
                x=[x_heard, x_used],
                y=[fy25_h[0], fy25_u[0]],
                mode='lines+markers',
                name=f'FY25' if i == 0 else '',
                line=dict(color='#8fc1e3', width=2, dash='dash'),
                marker=dict(size=8),
                showlegend=(i == 0),
                legendgroup='FY25',
                hovertemplate=f'{product}<br>%{{y:.1f}}%<extra></extra>'
            ))
        elif len(fy25_h) > 0 or len(fy25_u) > 0:
            # Plot individual points if only one exists (e.g., Po Chat has Heard but not Used in FY25)
            x_points = []
            y_points = []
            if len(fy25_h) > 0:
                x_points.append(x_heard)
                y_points.append(fy25_h[0])
            if len(fy25_u) > 0:
                x_points.append(x_used)
                y_points.append(fy25_u[0])
            fig_yetagon_comparison.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='markers',
                name=f'FY25' if i == 0 else '',
                marker=dict(color='#8fc1e3', size=8),
                showlegend=(i == 0),
                legendgroup='FY25',
                hovertemplate=f'{product}<br>%{{y:.1f}}%<extra></extra>'
            ))

    # Create x-axis with product groups
    # First layer: Heard Of / Used for each product
    # Second layer: Product names
    xaxis_dict = dict(
        tickmode='array',
        tickvals=all_tick_positions,
        ticktext=all_tick_labels,
        tickangle=0,
        showgrid=True
    )

    # Prepare annotations list with product labels
    annotations_list = []

    # Add product labels below the x-axis with smart wrapping (max 2 lines) and angled text
    for i, product in enumerate(products):
        # Smart wrapping: break into 2 lines if longer than ~15 chars
        words = product.split()
        if len(product) > 15 and len(words) > 1:
            # Split roughly in half (max 2 lines)
            mid = len(words) // 2
            product_wrapped = '<br>'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
        else:
            product_wrapped = product

        annotations_list.append(
            dict(
                x=product_positions[i],
                y=-0.15,
                text=f"<b>{product_wrapped}</b>",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=12),
                textangle=-45,
                xref="x",
                yref="paper"
            )
        )

    fig_yetagon_comparison.update_layout(
        xaxis=xaxis_dict,
        yaxis=dict(
            title='Percentage (%)'
        ),
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(b=120),  # Add bottom margin for product labels
        annotations=annotations_list
    )

    st.plotly_chart(fig_yetagon_comparison, use_container_width=True, key='yetagon_fy25_fy26_comparison')

    # Create slope chart for Farming Techniques
    st.markdown(f'#### Farming Techniques: FY25 vs FY26')

    technique_heard_labels = {
        'Q32_a': 'No Burn Rice Farming',
        'Q32_b': 'Salt Water Seed Selection',
        'Q32_c': 'Basal Fertilizer Usage for Rice',
        'Q32_d': 'Mid-season Fertilizer Usage for Rice',
        'Q32_e': 'Paddy Liming Acid',
        'Q32_f': 'Gypsum Application',
        'Q32_g': 'Boron Foliar Spray',
        'Q32_h': 'Epsom Salt Foliar Spray',
        'Q32_i': 'Neem Pesticide',
        'Q32_j': 'Fish Amino'
    }

    technique_used_labels = {
        'Q34_a': 'No Burn Rice Farming',
        'Q34_b': 'Salt Water Seed Selection',
        'Q34_c': 'Basal Fertilizer Usage for Rice',
        'Q34_d': 'Mid-season Fertilizer Usage for Rice',
        'Q34_e': 'Paddy Liming Acid',
        'Q34_f': 'Gypsum Application',
        'Q34_g': 'Boron Foliar Spray',
        'Q34_h': 'Epsom Salt Foliar Spray',
        'Q34_i': 'Neem Pesticide',
        'Q34_j': 'Fish Amino'
    }

    # Calculate percentages for FY25 and FY26 using filtered data
    fy26_tech_heard = calculate_year_percentages(prod_filtered, technique_heard_labels, technique_heard_labels)
    fy25_tech_heard = calculate_year_percentages(fy25_filtered, technique_heard_labels, technique_heard_labels)

    fy26_tech_used = calculate_year_percentages(prod_filtered, technique_used_labels, technique_used_labels)
    fy25_tech_used = calculate_year_percentages(fy25_filtered, technique_used_labels, technique_used_labels)

    # Create slope chart
    fig_technique_comparison = go.Figure()

    # Get unique techniques
    techniques = list(technique_heard_labels.values())
    n_techniques = len(techniques)

    # Create x-positions: each technique has 2 positions (Heard Of, Used)
    x_spacing = 3  # spacing between techniques

    all_tick_positions = []
    all_tick_labels = []
    technique_positions = []  # Center position for each technique label

    # Plot for each technique
    for i, technique in enumerate(techniques):
        # Calculate x positions for this technique
        x_heard = i * x_spacing
        x_used = i * x_spacing + 1

        all_tick_positions.extend([x_heard, x_used])
        all_tick_labels.extend(['Heard Of', 'Used'])
        technique_positions.append(i * x_spacing + 0.5)  # Center between heard and used

        # Get FY26 data
        fy26_h = fy26_tech_heard[fy26_tech_heard['Product'] == technique]['Percentage'].values
        fy26_u = fy26_tech_used[fy26_tech_used['Product'] == technique]['Percentage'].values

        # Get FY25 data
        fy25_h = fy25_tech_heard[fy25_tech_heard['Product'] == technique]['Percentage'].values
        fy25_u = fy25_tech_used[fy25_tech_used['Product'] == technique]['Percentage'].values

        # FY26 line (Heard Of to Used)
        if len(fy26_h) > 0 and len(fy26_u) > 0:
            fig_technique_comparison.add_trace(go.Scatter(
                x=[x_heard, x_used],
                y=[fy26_h[0], fy26_u[0]],
                mode='lines+markers',
                name=f'FY26' if i == 0 else '',
                line=dict(color='#d73027', width=2),
                marker=dict(size=8),
                showlegend=(i == 0),
                legendgroup='FY26',
                hovertemplate=f'{technique}<br>%{{y:.1f}}%<extra></extra>'
            ))
        elif len(fy26_h) > 0 or len(fy26_u) > 0:
            # Plot individual points if only one exists
            x_points = []
            y_points = []
            if len(fy26_h) > 0:
                x_points.append(x_heard)
                y_points.append(fy26_h[0])
            if len(fy26_u) > 0:
                x_points.append(x_used)
                y_points.append(fy26_u[0])
            fig_technique_comparison.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='markers',
                name=f'FY26' if i == 0 else '',
                marker=dict(color='#d73027', size=8),
                showlegend=(i == 0),
                legendgroup='FY26',
                hovertemplate=f'{technique}<br>%{{y:.1f}}%<extra></extra>'
            ))

        # FY25 line (Heard Of to Used)
        if len(fy25_h) > 0 and len(fy25_u) > 0:
            fig_technique_comparison.add_trace(go.Scatter(
                x=[x_heard, x_used],
                y=[fy25_h[0], fy25_u[0]],
                mode='lines+markers',
                name=f'FY25' if i == 0 else '',
                line=dict(color='#f39c12', width=2, dash='dash'),
                marker=dict(size=8),
                showlegend=(i == 0),
                legendgroup='FY25',
                hovertemplate=f'{technique}<br>%{{y:.1f}}%<extra></extra>'
            ))
        elif len(fy25_h) > 0 or len(fy25_u) > 0:
            # Plot individual points if only one exists
            x_points = []
            y_points = []
            if len(fy25_h) > 0:
                x_points.append(x_heard)
                y_points.append(fy25_h[0])
            if len(fy25_u) > 0:
                x_points.append(x_used)
                y_points.append(fy25_u[0])
            fig_technique_comparison.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='markers',
                name=f'FY25' if i == 0 else '',
                marker=dict(color='#f39c12', size=8),
                showlegend=(i == 0),
                legendgroup='FY25',
                hovertemplate=f'{technique}<br>%{{y:.1f}}%<extra></extra>'
            ))

    # Create x-axis with technique groups
    xaxis_dict = dict(
        tickmode='array',
        tickvals=all_tick_positions,
        ticktext=all_tick_labels,
        tickangle=0,
        showgrid=True
    )

    # Prepare annotations list with technique labels
    technique_annotations_list = []

    # Add technique labels below the x-axis with smart wrapping (max 2 lines) and angled text
    for i, technique in enumerate(techniques):
        # Smart wrapping: break into 2 lines if longer than ~15 chars
        words = technique.split()
        if len(technique) > 15 and len(words) > 1:
            # Split roughly in half (max 2 lines)
            mid = len(words) // 2
            technique_wrapped = '<br>'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
        else:
            technique_wrapped = technique

        technique_annotations_list.append(
            dict(
                x=technique_positions[i],
                y=-0.15,
                text=f"<b>{technique_wrapped}</b>",
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=12),
                textangle=-45,
                xref="x",
                yref="paper"
            )
        )

    fig_technique_comparison.update_layout(
        xaxis=xaxis_dict,
        yaxis=dict(
            title='Percentage (%)'
        ),
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(b=140),  # Add bottom margin for technique labels
        annotations=technique_annotations_list
    )

    st.plotly_chart(fig_technique_comparison, use_container_width=True, key='technique_fy25_fy26_comparison')

    st.markdown('---')

    # Paid vs Free Products Usage Comparison Section
    st.subheader('Paid vs Free Products Usage Comparison')
    st.caption('Note: Paid products include Yetagon Irrigation, EM/Zarmani, Trichoderma, Sun-kissed, and Fish Amino. Free products include Digital Products (Q34 techniques, Po Chat, Messenger, Digital farm practices) and Yetagon Tele-agronomy.')

    # Define paid product columns (used)
    paid_product_cols = ['Q33_a', 'Q33_b', 'Q33_c', 'Q33_h', 'Q33_i']  # Irrigation, EM/Zarmani, Trichoderma, Sun-kissed, Fish Amino
    paid_product_names = {
        'Q33_a': 'Yetagon Irrigation',
        'Q33_b': 'Yetagon EM/Zarmani',
        'Q33_c': 'Yetagon Trichoderma',
        'Q33_h': 'Yetagon Sun-kissed',
        'Q33_i': 'Yetagon Fish Amino'
    }

    # Define free product columns (used) - Digital Products + Tele-agronomy
    # Digital products: Q34_a through Q34_k (techniques) + Q33_j (Po Chat) + Q33_k (Messenger) + Q33_e (Digital farm practices)
    # Plus Yetagon Tele-agronomy: Q33_d
    free_product_technique_cols = [f'Q34_{chr(97+i)}' for i in range(11)]  # Q34_a through Q34_k
    free_product_other_cols = ['Q33_j', 'Q33_k', 'Q33_e', 'Q33_d']  # Po Chat, Messenger, Digital farm practices, Tele-agronomy
    free_product_cols = free_product_technique_cols + free_product_other_cols

    # Calculate average usage for paid products
    paid_usage_data = prod_filtered.copy()
    paid_usage_count = 0
    paid_products_available = 0
    for col in paid_product_cols:
        if col in paid_usage_data.columns:
            paid_usage_count += paid_usage_data[col].fillna(0).astype(int).sum()
            paid_products_available += 1

    # Calculate average usage for free products
    free_usage_data = prod_filtered.copy()
    free_usage_count = 0
    free_products_available = 0
    for col in free_product_cols:
        if col in free_usage_data.columns:
            free_usage_count += free_usage_data[col].fillna(0).astype(int).sum()
            free_products_available += 1

    total_respondents = len(prod_filtered)

    # Calculate average number of products used per respondent
    if total_respondents > 0:
        avg_paid_usage = paid_usage_count / total_respondents
        avg_free_usage = free_usage_count / total_respondents
    else:
        avg_paid_usage = 0
        avg_free_usage = 0

    # Create two columns for display
    paid_free_col1, paid_free_col2 = st.columns(2)

    with paid_free_col1:
        st.markdown('#### Average Products Used per Farmer')

        # Create bar chart comparing average usage
        avg_usage_df = pd.DataFrame({
            'Product Type': ['Paid Products', 'Free Products'],
            'Average Used': [avg_paid_usage, avg_free_usage]
        })

        fig_avg_usage = go.Figure()
        fig_avg_usage.add_trace(go.Bar(
            x=avg_usage_df['Product Type'],
            y=avg_usage_df['Average Used'],
            marker_color=['#0f4c3a', '#8fc1e3'],
            text=[f'{val:.2f}' for val in avg_usage_df['Average Used']],
            textposition='outside',
            hovertemplate='%{x}<br>Average: %{y:.2f} products<extra></extra>'
        ))

        fig_avg_usage.update_layout(
            xaxis_title='',
            yaxis_title='Average Number of Products Used',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_avg_usage, use_container_width=True, key='avg_paid_free_usage_chart')

    with paid_free_col2:
        st.markdown('#### Usage Breakdown by Product')

        # Calculate individual product usage percentages
        product_usage_list = []

        # Paid products
        for col in paid_product_cols:
            if col in prod_filtered.columns:
                usage_pct = (prod_filtered[col].fillna(0).astype(int).sum() / total_respondents * 100) if total_respondents > 0 else 0
                product_usage_list.append({
                    'Product': paid_product_names.get(col, col),
                    'Usage (%)': usage_pct,
                    'Type': 'Paid'
                })

        # Free products - show aggregated for digital techniques and individual for others
        # Digital techniques aggregated
        digital_tech_usage = 0
        for col in free_product_technique_cols:
            if col in prod_filtered.columns:
                digital_tech_usage += prod_filtered[col].fillna(0).astype(int).sum()
        # Count users who used any digital technique
        digital_tech_users = prod_filtered.copy()
        digital_tech_users['Any_Digital_Tech'] = False
        for col in free_product_technique_cols:
            if col in digital_tech_users.columns:
                digital_tech_users['Any_Digital_Tech'] = digital_tech_users['Any_Digital_Tech'] | (digital_tech_users[col] == 1.0)
        digital_tech_pct = (digital_tech_users['Any_Digital_Tech'].sum() / total_respondents * 100) if total_respondents > 0 else 0
        product_usage_list.append({
            'Product': 'Digital Techniques (Any)',
            'Usage (%)': digital_tech_pct,
            'Type': 'Free'
        })

        # Individual free products
        free_product_names = {
            'Q33_j': 'Po Chat',
            'Q33_k': 'Messenger',
            'Q33_e': 'Digital Farm Practices',
            'Q33_d': 'Yetagon Tele-agronomy'
        }
        for col in free_product_other_cols:
            if col in prod_filtered.columns:
                usage_pct = (prod_filtered[col].fillna(0).astype(int).sum() / total_respondents * 100) if total_respondents > 0 else 0
                product_usage_list.append({
                    'Product': free_product_names.get(col, col),
                    'Usage (%)': usage_pct,
                    'Type': 'Free'
                })

        if product_usage_list:
            product_usage_df = pd.DataFrame(product_usage_list)
            product_usage_df = product_usage_df.sort_values(['Type', 'Usage (%)'], ascending=[True, False])

            fig_product_breakdown = px.bar(
                product_usage_df,
                x='Product',
                y='Usage (%)',
                color='Type',
                color_discrete_map={'Paid': '#0f4c3a', 'Free': '#8fc1e3'},
                barmode='group'
            )
            fig_product_breakdown.update_layout(
                xaxis_title='',
                yaxis_title='Usage (%)',
                height=400,
                xaxis={'tickangle': -45},
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            st.plotly_chart(fig_product_breakdown, use_container_width=True, key='product_usage_breakdown_chart')

    st.markdown('---')

    # NPS Analysis Section
    st.subheader('NPS Analysis')

    # Calculate NPS Score
    if 'Q40' in prod_filtered.columns and 'Q41' in prod_filtered.columns:
        valid_scores = prod_filtered['Q40'].dropna()

        if len(valid_scores) > 0:
            # Create two columns for NPS metrics
            nps_col1, nps_col2 = st.columns(2)

            with nps_col1:
                # NPS Distribution Chart
                st.markdown('#### NPS Distribution Chart')

                # Create distribution data
                score_distribution = valid_scores.value_counts().sort_index().reset_index()
                score_distribution.columns = ['NPS Rating', 'Count']

                # Ensure all scores 0-10 are present
                all_scores = pd.DataFrame({'NPS Rating': range(0, 11)})
                score_distribution = all_scores.merge(score_distribution, on='NPS Rating', how='left').fillna(0)
                score_distribution['Count'] = score_distribution['Count'].astype(int)
                score_distribution['Survey'] = 'Current Survey'

                fig_nps_dist = px.bar(
                    score_distribution,
                    x='NPS Rating',
                    y='Count',
                    color='Survey',
                    color_discrete_map={'Current Survey': '#0f4c3a', 'Previous Survey': '#a8d5c7'},
                    barmode='group'
                )
                fig_nps_dist.update_layout(
                    showlegend=True,
                    xaxis_title='NPS Rating (0-10 scale)',
                    yaxis_title='',
                    height=400,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1)
                )
                st.plotly_chart(fig_nps_dist, use_container_width=True, key='nps_distribution_chart')

            with nps_col2:
                # Reasons by Product
                st.markdown('#### Reasons by Product')

                # Define product columns
                product_columns = {
                    'Yetagon Irrigation': 'Q33_a',
                    'Yetagon EM/Zarmani': 'Q33_b',
                    'Yetagon Trichoderma': 'Q33_c',
                    'Yetagon Tele-agronomy': 'Q33_d',
                    'Digital farm practices': 'Q33_e',
                    'Yetagon Sun-kissed': 'Q33_h',
                    'Yetagon Fish Amino': 'Q33_i',
                    'Po Chat': 'Q33_j',
                    'Messenger': 'Q33_k'
                }

                # Create reasons breakdown by product with NPS scores
                reasons_by_product = []
                for product_name, product_col in product_columns.items():
                    if product_col in prod_filtered.columns:
                        product_users = prod_filtered[prod_filtered[product_col] == 1.0]

                        if len(product_users) > 0 and 'Q41' in product_users.columns and 'Q40' in product_users.columns:
                            # Iterate through each user's response
                            for idx, row in product_users.iterrows():
                                if pd.notna(row['Q41']) and pd.notna(row['Q40']):
                                    reasons_by_product.append({
                                        'Product': product_name,
                                        'NPS Score': int(row['Q40']),
                                        'Reason': str(row['Q41'])[:50] + ('...' if len(str(row['Q41'])) > 50 else '')
                                    })

                if reasons_by_product:
                    reasons_product_df = pd.DataFrame(reasons_by_product)

                    # Sort by product and NPS score
                    reasons_product_df = reasons_product_df.sort_values(['Product', 'NPS Score'], ascending=[True, False])

                    # Display as scrollable dataframe
                    st.dataframe(
                        reasons_product_df,
                        hide_index=True,
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info('No reason data available')
    else:
        st.info('NPS data not available in the dataset')

    st.markdown('---')

    # NPS Categories by Product Used Section
    st.subheader('NPS Categories by Product Used')

    # Calculate NPS categories
    def calculate_nps_category(score):
        if pd.isna(score):
            return None
        if score >= 0 and score <= 6:
            return 'Detractor'
        elif score >= 7 and score <= 8:
            return 'Passive'
        elif score >= 9 and score <= 10:
            return 'Promoter'
        return None

    # Add NPS category to filtered data
    nps_data = prod_filtered.copy()
    if 'Q40' in nps_data.columns:
        nps_data['NPS_Category'] = nps_data['Q40'].apply(calculate_nps_category)

        # Define products to analyze
        product_columns = {
            'Yetagon Irrigation': 'Q33_a',
            'Yetagon EM/Zarmani': 'Q33_b',
            'Yetagon Trichoderma': 'Q33_c',
            'Yetagon Tele-agronomy': 'Q33_d',
            'Digital farm practices': 'Q33_e',
            'Yetagon Sun-kissed': 'Q33_h',
            'Yetagon Fish Amino': 'Q33_i',
            'Po Chat': 'Q33_j',
            'Messenger': 'Q33_k'
        }

        # Calculate NPS breakdown by product
        nps_results = []
        for product_name, product_col in product_columns.items():
            if product_col in nps_data.columns:
                product_users = nps_data[nps_data[product_col] == 1.0]

                if len(product_users) > 0:
                    detractors = (product_users['NPS_Category'] == 'Detractor').sum()
                    passives = (product_users['NPS_Category'] == 'Passive').sum()
                    promoters = (product_users['NPS_Category'] == 'Promoter').sum()
                    total = detractors + passives + promoters

                    nps_results.append({
                        'Product': product_name,
                        'Detractor': detractors,
                        'Passive': passives,
                        'Promoter': promoters,
                        'Total': total
                    })

        nps_results_df = pd.DataFrame(nps_results)

        if not nps_results_df.empty:
            st.markdown('#### NPS Distribution by Product')

            # Create two columns for chart and table
            nps_col1, nps_col2 = st.columns([1, 1])

            with nps_col1:
                # Create a stacked bar chart
                nps_chart_data = nps_results_df.set_index('Product')[['Detractor', 'Passive', 'Promoter']]

                fig_nps = px.bar(
                    nps_chart_data,
                    x=nps_chart_data.index,
                    y=['Detractor', 'Passive', 'Promoter'],
                    barmode='stack',
                    color_discrete_map={
                        'Detractor': '#e74c3c',
                        'Passive': '#f39c12',
                        'Promoter': '#27ae60'
                    },
                    labels={'value': 'Count', 'variable': 'NPS Category'}
                )
                fig_nps.update_layout(xaxis_title='', yaxis_title='Count', height=500)
                st.plotly_chart(fig_nps, use_container_width=True, key='nps_by_product_chart')

            with nps_col2:
                # Create detailed NPS table with score counts and average
                st.markdown('**NPS Score Details**')

                # Calculate score counts and averages for each product
                nps_table_data = []
                for product_name, product_col in product_columns.items():
                    if product_col in nps_data.columns:
                        product_users = nps_data[nps_data[product_col] == 1.0]

                        if len(product_users) > 0 and 'Q40' in product_users.columns:
                            # Count each score from 0-10
                            score_counts = {}
                            for score in range(11):
                                score_counts[str(score)] = (product_users['Q40'] == score).sum()

                            # Calculate average
                            avg_score = product_users['Q40'].mean()

                            row_data = {'Product': product_name}
                            row_data.update(score_counts)
                            row_data['Average'] = avg_score
                            nps_table_data.append(row_data)

                nps_table_df = pd.DataFrame(nps_table_data)

                if not nps_table_df.empty:
                    # Format the average column to 2 decimal places
                    nps_table_df['Average'] = nps_table_df['Average'].apply(lambda x: f'{x:.2f}')

                    # Display the table
                    st.dataframe(
                        nps_table_df,
                        use_container_width=True,
                        hide_index=True,
                        height=500
                    )
    else:
        st.info('NPS data (Q40) not available in the dataset.')

with tab5:
    st.header('Location Impact Analysis')

    # Load township mapping CSVs for each product
    @st.cache_data
    def load_township_mappings():
        mighty_df = pd.read_csv('mighty.csv')
        pheonix_df = pd.read_csv('pheonix.csv')
        sun_kissed_df = pd.read_csv('sun_kissed.csv')

        # Extract township lists for each product
        mighty_both = mighty_df['Both'].dropna().str.strip().tolist()
        mighty_digital = mighty_df['Direct'].dropna().str.strip().tolist()  # "Direct" column contains Digital-only townships

        pheonix_both = pheonix_df['Both'].dropna().str.strip().tolist()
        pheonix_digital = pheonix_df['Direct'].dropna().str.strip().tolist()

        sun_kissed_both = sun_kissed_df['Both'].dropna().str.strip().tolist()
        sun_kissed_digital = sun_kissed_df['Direct'].dropna().str.strip().tolist()

        return {
            'mighty': {'both': mighty_both, 'digital': mighty_digital},
            'pheonix': {'both': pheonix_both, 'digital': pheonix_digital},
            'sun_kissed': {'both': sun_kissed_both, 'digital': sun_kissed_digital}
        }

    township_mappings = load_township_mappings()

    # Product definitions
    product_info = {
        'Phoenix (EM/Zarmani)': {
            'heard_col': 'Q31_b',
            'used_col': 'Q33_b',
            'townships': township_mappings['pheonix']
        },
        'Mighty (Trichoderma/Barhmati)': {
            'heard_col': 'Q31_c',
            'used_col': 'Q33_c',
            'townships': township_mappings['mighty']
        },
        'Sun-kissed': {
            'heard_col': 'Q31_e',
            'used_col': 'Q33_h',
            'townships': township_mappings['sun_kissed']
        }
    }

    # Use full dataset for this analysis
    df_tab5 = df.copy()

    st.markdown(f"**Sample Size: n = {len(df_tab5)}**")
    st.markdown("---")

    # Helper function to classify township and calculate metrics
    def calculate_location_metrics(df_filtered, product_config):
        """Calculate awareness metrics for Both vs Digital-only townships"""
        both_townships = product_config['townships']['both']
        digital_townships = product_config['townships']['digital']

        heard_col = product_config['heard_col']
        used_col = product_config['used_col']

        # Filter by township groups
        df_both = df_filtered[df_filtered['Township'].isin(both_townships)]
        df_digital = df_filtered[df_filtered['Township'].isin(digital_townships)]

        results = []

        # Calculate metrics for "Both" group
        if len(df_both) > 0:
            heard_both_count = (df_both[heard_col] == 1.0).sum() if heard_col in df_both.columns else 0
            used_both_count = (df_both[used_col] == 1.0).sum() if used_col in df_both.columns else 0
            heard_both_pct = heard_both_count / len(df_both) * 100 if len(df_both) > 0 else 0
            used_both_pct = used_both_count / len(df_both) * 100 if len(df_both) > 0 else 0
            results.append({
                'Group': 'Both (Direct + Digital)',
                'Sample Size': len(df_both),
                'Heard Of (%)': heard_both_pct,
                'Used (%)': used_both_pct,
                'Heard Of (Count)': heard_both_count,
                'Used (Count)': used_both_count
            })

        # Calculate metrics for "Digital" group
        if len(df_digital) > 0:
            heard_digital_count = (df_digital[heard_col] == 1.0).sum() if heard_col in df_digital.columns else 0
            used_digital_count = (df_digital[used_col] == 1.0).sum() if used_col in df_digital.columns else 0
            heard_digital_pct = heard_digital_count / len(df_digital) * 100 if len(df_digital) > 0 else 0
            used_digital_pct = used_digital_count / len(df_digital) * 100 if len(df_digital) > 0 else 0
            results.append({
                'Group': 'Digital Only',
                'Sample Size': len(df_digital),
                'Heard Of (%)': heard_digital_pct,
                'Used (%)': used_digital_pct,
                'Heard Of (Count)': heard_digital_count,
                'Used (Count)': used_digital_count
            })

        return pd.DataFrame(results), df_both, df_digital

    # ========== OVERVIEW CHART: All 3 Products ==========
    st.subheader('Product Awareness & Usage: All Products Overview')
    st.caption('Percentages are calculated within each respective population (Both or Digital Only)')

    # Calculate metrics for all products
    all_products_data = []
    for prod_name, prod_config in product_info.items():
        metrics_df_temp, _, _ = calculate_location_metrics(df_tab5, prod_config)

        if not metrics_df_temp.empty:
            for _, row in metrics_df_temp.iterrows():
                all_products_data.append({
                    'Product': prod_name.replace(' (EM/Zarmani)', '').replace(' (Trichoderma/Barhmati)', ''),
                    'Group': row['Group'],
                    'Heard Of (%)': row['Heard Of (%)'],
                    'Used (%)': row['Used (%)'],
                    'Heard Of (Count)': row['Heard Of (Count)'],
                    'Used (Count)': row['Used (Count)'],
                    'Sample Size': row['Sample Size']
                })

    all_products_df = pd.DataFrame(all_products_data)

    if not all_products_df.empty:
        # Create grouped bar chart with all products - grouped by product (using percentages)
        fig_overview = go.Figure()

        products_list = all_products_df['Product'].unique()

        # Different color schemes per product (dark for Both, light for Digital)
        product_colors = {
            'Mighty': {'both': '#0f4c3a', 'digital': '#5a8f7b'},      # Green shades
            'Phoenix': {'both': '#1565c0', 'digital': '#64b5f6'},     # Blue shades
            'Sun-kissed': {'both': '#e65100', 'digital': '#ffb74d'}   # Orange shades
        }

        # Build x-axis categories: Product + Metric combinations
        x_categories = []
        for product in products_list:
            x_categories.append(f'{product}<br>Heard Of')
            x_categories.append(f'{product}<br>Used')

        # Prepare data and colors for each location group (percentages)
        both_values = []
        digital_values = []
        both_colors = []
        digital_colors = []

        for product in products_list:
            prod_data = all_products_df[all_products_df['Product'] == product]
            both_row = prod_data[prod_data['Group'] == 'Both (Direct + Digital)']
            digital_row = prod_data[prod_data['Group'] == 'Digital Only']

            colors = product_colors.get(product, {'both': '#0f4c3a', 'digital': '#5a8f7b'})

            # Heard Of (%)
            both_values.append(both_row['Heard Of (%)'].values[0] if len(both_row) > 0 else 0)
            digital_values.append(digital_row['Heard Of (%)'].values[0] if len(digital_row) > 0 else 0)
            both_colors.append(colors['both'])
            digital_colors.append(colors['digital'])

            # Used (%)
            both_values.append(both_row['Used (%)'].values[0] if len(both_row) > 0 else 0)
            digital_values.append(digital_row['Used (%)'].values[0] if len(digital_row) > 0 else 0)
            both_colors.append(colors['both'])
            digital_colors.append(colors['digital'])

        # Add bars for Both (Direct + Digital)
        fig_overview.add_trace(go.Bar(
            name='Both (Direct + Digital)',
            x=x_categories,
            y=both_values,
            text=[f"{v:.1f}%" for v in both_values],
            textposition='outside',
            marker_color=both_colors
        ))

        # Add bars for Digital Only
        fig_overview.add_trace(go.Bar(
            name='Digital Only',
            x=x_categories,
            y=digital_values,
            text=[f"{v:.1f}%" for v in digital_values],
            textposition='outside',
            marker_color=digital_colors
        ))

        fig_overview.update_layout(
            barmode='group',
            xaxis_title='',
            yaxis_title='Percentage (%)',
            height=450,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_overview, use_container_width=True, key='overview_all_products_chart')

        # Display sample sizes for reference
        st.caption('Sample sizes by product and location type:')
        sample_sizes = all_products_df[['Product', 'Group', 'Sample Size']].drop_duplicates()
        sample_sizes_pivot = sample_sizes.pivot(index='Product', columns='Group', values='Sample Size').reset_index()
        st.dataframe(sample_sizes_pivot, hide_index=True, use_container_width=True)

    st.markdown('---')

    # ========== DETAILED ANALYSIS: Selected Product ==========
    st.subheader('Detailed Analysis by Product')

    # Product selector using dropdown
    selected_product = st.selectbox(
        'Select Product for Detailed Analysis',
        list(product_info.keys()),
        key='tab5_product_select'
    )

    product_config = product_info[selected_product]
    metrics_df, df_both, df_digital = calculate_location_metrics(df_tab5, product_config)

    if not metrics_df.empty:
        # Expanded Analysis: Other Yetagon Products Awareness
        st.subheader('Other Yetagon Products Awareness by Location Type')

        other_products = {
            'Q31_a': 'Yetagon Irrigation',
            'Q31_b': 'Yetagon EM/Zarmani',
            'Q31_c': 'Yetagon Trichoderma/Barhmati',
            'Q31_d': 'Yetagon Tele-agronomy',
            'Q31_e': 'Yetagon Sun-kissed',
            'Q31_f': 'Yetagon Fish Amino',
            'Q31_g': 'Po Chat',
            'Q31_h': 'Messenger',
            'Q31_i': 'Digital farm practices'
        }

        other_products_data = []

        for col, label in other_products.items():
            both_pct = (df_both[col] == 1.0).sum() / len(df_both) * 100 if col in df_both.columns and len(df_both) > 0 else 0
            digital_pct = (df_digital[col] == 1.0).sum() / len(df_digital) * 100 if col in df_digital.columns and len(df_digital) > 0 else 0

            other_products_data.append({
                'Product': label,
                'Both (Direct + Digital)': both_pct,
                'Digital Only': digital_pct,
                'Difference': both_pct - digital_pct
            })

        other_products_df = pd.DataFrame(other_products_data)

        # Create grouped bar chart for other products (percentages)
        fig_other_products = go.Figure()

        fig_other_products.add_trace(go.Bar(
            name='Both (Direct + Digital)',
            x=other_products_df['Product'],
            y=other_products_df['Both (Direct + Digital)'],
            text=[f"{v:.1f}%" for v in other_products_df['Both (Direct + Digital)']],
            textposition='outside',
            marker_color='#0f4c3a'
        ))

        fig_other_products.add_trace(go.Bar(
            name='Digital Only',
            x=other_products_df['Product'],
            y=other_products_df['Digital Only'],
            text=[f"{v:.1f}%" for v in other_products_df['Digital Only']],
            textposition='outside',
            marker_color='#8fc1e3'
        ))

        fig_other_products.update_layout(
            barmode='group',
            xaxis_title='',
            yaxis_title='Percentage (%)',
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis={'tickangle': -45}
        )
        st.plotly_chart(fig_other_products, use_container_width=True, key='other_products_location_chart')

        st.markdown('---')

        # Brand Awareness by Location Type
        st.subheader('Brand Awareness by Location Type')

        brand_labels = {
            'Q36_a': 'Awba',
            'Q36_b': 'Golden Lion',
            'Q36_c': 'Armo',
            'Q36_d': 'MWD-Shan Maw',
            'Q36_e': 'Farmer Phoe Wa',
            'Q36_f': 'Pyi Htaung Su',
            'Q36_g': 'Great Tiger',
            'Q36_h': 'Hatake',
            'Q36_i': 'Ywat Lwint',
            'Q36_j': 'Pyan Lwar',
            'Q36_k': 'Doh Kyay Let',
            'Q36_l': 'Eco-Vital Bio-stimulant',
            'Q36_m': 'Other'
        }

        brand_data = []

        for col, label in brand_labels.items():
            both_pct = (df_both[col] == 1.0).sum() / len(df_both) * 100 if col in df_both.columns and len(df_both) > 0 else 0
            digital_pct = (df_digital[col] == 1.0).sum() / len(df_digital) * 100 if col in df_digital.columns and len(df_digital) > 0 else 0

            brand_data.append({
                'Brand': label,
                'Both (Direct + Digital)': both_pct,
                'Digital Only': digital_pct,
                'Difference': both_pct - digital_pct
            })

        brand_df = pd.DataFrame(brand_data)

        # Sort by average awareness
        brand_df['Average'] = (brand_df['Both (Direct + Digital)'] + brand_df['Digital Only']) / 2
        brand_df = brand_df.sort_values('Average', ascending=False)

        # Create grouped bar chart for brands (percentages)
        fig_brands = go.Figure()

        fig_brands.add_trace(go.Bar(
            name='Both (Direct + Digital)',
            x=brand_df['Brand'],
            y=brand_df['Both (Direct + Digital)'],
            text=[f"{v:.1f}%" for v in brand_df['Both (Direct + Digital)']],
            textposition='outside',
            marker_color='#0f4c3a'
        ))

        fig_brands.add_trace(go.Bar(
            name='Digital Only',
            x=brand_df['Brand'],
            y=brand_df['Digital Only'],
            text=[f"{v:.1f}%" for v in brand_df['Digital Only']],
            textposition='outside',
            marker_color='#8fc1e3'
        ))

        fig_brands.update_layout(
            barmode='group',
            xaxis_title='',
            yaxis_title='Percentage (%)',
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis={'tickangle': -45}
        )
        st.plotly_chart(fig_brands, use_container_width=True, key='brands_location_chart')

        st.markdown('---')

        # Summary Table
        st.subheader('Summary: Difference in Awareness (Both - Digital Only)')

        # Combine products and brands into summary
        summary_data = []

        for _, row in other_products_df.iterrows():
            summary_data.append({
                'Category': 'Yetagon Product',
                'Item': row['Product'],
                'Both (%)': row['Both (Direct + Digital)'],
                'Digital Only (%)': row['Digital Only'],
                'Difference (pp)': row['Difference']
            })

        for _, row in brand_df.iterrows():
            summary_data.append({
                'Category': 'Brand',
                'Item': row['Brand'],
                'Both (%)': row['Both (Direct + Digital)'],
                'Digital Only (%)': row['Digital Only'],
                'Difference (pp)': row['Difference']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Difference (pp)', ascending=False)

        # Format the dataframe for display
        summary_display = summary_df.copy()
        summary_display['Both (%)'] = summary_display['Both (%)'].apply(lambda x: f"{x:.1f}%")
        summary_display['Digital Only (%)'] = summary_display['Digital Only (%)'].apply(lambda x: f"{x:.1f}%")
        summary_display['Difference (pp)'] = summary_display['Difference (pp)'].apply(lambda x: f"{x:+.1f}")

        st.dataframe(summary_display, hide_index=True, use_container_width=True)

    else:
        st.warning('No data available for the selected product and filters. Please check if townships in the mapping files match the survey data.')
