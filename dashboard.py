import streamlit as st
import pandas as pd
import plotly.express as px
from platform_helper import create_platform_analysis

# Page config
st.set_page_config(page_title='Survey Demographics Dashboard', layout='wide')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned.csv')
    questions_df = pd.read_csv('questions.csv')
    return df, questions_df

df, questions_df = load_data()

# Title
st.title('📊 Survey Demographics Dashboard')

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(['Demographics', 'Digital Behavior & Engagement', 'Internet Usage Behaviors', 'Brand & Product Awareness'])

with tab1:
    # Demographics metrics
    st.header('Demographics Overview')
    col1, col2, col3, col4, col5 = st.columns(5)

    # Total respondents
    col1.metric('Total Respondents', len(df))

    # Direct percentage
    if 'Direct or Digital?' in df.columns:
        direct_count = (df['Direct or Digital?'] == 'Direct').sum()
        direct_pct = (direct_count / len(df) * 100) if len(df) > 0 else 0
        col2.metric('Direct', f"{direct_pct:.1f}%")
    else:
        col2.metric('Direct', 'N/A')

    # Male percentage
    if 'Q03' in df.columns:
        male_count = (df['Q03'] == 'Male').sum()
        male_pct = (male_count / len(df) * 100) if len(df) > 0 else 0
        col3.metric('Male', f"{male_pct:.1f}%")
    else:
        col3.metric('Male', 'N/A')

    # Heard of Po Chat percentage
    if 'Q31_g' in df.columns:
        po_chat_count = (df['Q31_g'] == 1.0).sum()
        po_chat_pct = (po_chat_count / len(df) * 100) if len(df) > 0 else 0
        col4.metric('Heard Po Chat', f"{po_chat_pct:.1f}%")
    else:
        col4.metric('Heard Po Chat', 'N/A')

    # Average NPS Score
    if 'Q40' in df.columns:
        nps_avg = df['Q40'].mean()
        col5.metric('Avg NPS Score', f"{nps_avg:.2f}")
    else:
        col5.metric('Avg NPS Score', 'N/A')

    # First row: Regional Distribution (pie) and Age Distribution (bar)
    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:
        st.subheader('Regional Distribution')
        region_counts = df['Region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']

        fig_region = px.pie(
            region_counts,
            names='Region',
            values='Count',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_region.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
        st.plotly_chart(fig_region, use_container_width=True, key='region_pie_chart')

    with demo_col2:
        st.subheader('Age Distribution')
        if 'Q02' in df.columns:
            age_counts = df['Q02'].value_counts().reset_index()
            age_counts.columns = ['Age Group', 'Count']
            # Sort by age order
            age_order = ['Under 20', '20-30', '31-40', '41-50', '51-60', 'Over 60']
            age_counts['Age Group'] = pd.Categorical(age_counts['Age Group'], categories=age_order, ordered=True)
            age_counts = age_counts.sort_values('Age Group')
            age_counts['Percentage'] = (age_counts['Count'] / age_counts['Count'].sum() * 100).round(1)

            fig_age = px.bar(
                age_counts,
                x='Age Group',
                y='Count',
                text='Percentage',
                color='Count',
                color_continuous_scale='Oranges'
            )
            fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_age.update_layout(showlegend=False, xaxis_title='', yaxis_title='')
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

        # Prepare data for UpSet plot
        q09_cols = [col for col in q09_labels.keys() if col in df.columns]

        # Create a DataFrame for upset plot - need to convert to boolean
        upset_data = df[q09_cols].copy()
        upset_data = upset_data.fillna(0)
        upset_data = (upset_data == 1.0)

        # Rename columns to friendly names
        rename_map = {col: q09_labels[col] for col in q09_cols if col in q09_labels}
        upset_data = upset_data.rename(columns=rename_map)

        # Count combinations
        from collections import Counter
        combinations = []
        for idx, row in upset_data.iterrows():
            combo = tuple(col for col in upset_data.columns if row[col])
            if combo:  # Only include if at least one source is selected
                combinations.append(combo)

        combo_counts = Counter(combinations)

        # Get top 15 combinations
        top_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[:15]

        # Create upset-style visualization manually
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Prepare data
        combo_labels = []
        combo_values = []
        combo_percentages = []
        matrix_data = {col: [] for col in upset_data.columns}

        total_respondents = len(df)

        for combo, count in top_combos:
            combo_labels.append(' & '.join(combo) if len(combo) <= 2 else f"{len(combo)} sources")
            combo_values.append(count)
            pct = (count / total_respondents * 100) if total_respondents > 0 else 0
            combo_percentages.append(pct)

            for col in upset_data.columns:
                matrix_data[col].append(1 if col in combo else 0)

        # Create figure with subplots
        fig_q09 = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.1,
            specs=[[{"type": "bar"}], [{"type": "scatter"}]]
        )

        # Top plot: bar chart of combination percentages
        fig_q09.add_trace(
            go.Bar(
                x=list(range(len(combo_percentages))),
                y=combo_percentages,
                marker_color='#4575b4',
                showlegend=False,
                width=0.6,
                text=[f'{p:.1f}%' for p in combo_percentages],
                textposition='outside',
                hovertemplate='%{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Bottom plot: matrix showing which sources are in each combination
        # Add connecting lines first
        for j in range(len(matrix_data[list(upset_data.columns)[0]])):
            # Find which sources are active for this combination
            active_indices = [i for i, col in enumerate(upset_data.columns) if matrix_data[col][j] == 1]

            if len(active_indices) > 1:
                # Draw connecting line
                fig_q09.add_trace(
                    go.Scatter(
                        x=[j] * len(active_indices),
                        y=active_indices,
                        mode='lines',
                        line=dict(color='#d73027', width=2),
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
                    marker=dict(size=10, color='#d73027', line=dict(color='#ffffff', width=1)),
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
            range=[-0.5, len(combo_values) - 0.5],
            row=1, col=1
        )
        fig_q09.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, len(combo_values) - 0.5],
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
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_q09, use_container_width=True, key='farming_info_upset')

    with demo_col4:
        st.subheader('Mobile Top-up Amount per Week')
        if 'Q29' in df.columns:
            q29_counts = df['Q29'].value_counts().reset_index()
            q29_counts.columns = ['Top-up Amount', 'Count']

            # Standardize labels
            label_mapping = {
                'Less than 3000 MMK': '<3000 MMK',
                'less than 3000 MMK': '<3000 MMK',
                'approximately 5000 MMK': '~5000 MMK',
                'above 10000 MMK': '>10000 MMK',
                'other': 'Other'
            }
            q29_counts['Top-up Amount'] = q29_counts['Top-up Amount'].map(label_mapping).fillna(q29_counts['Top-up Amount'])

            fig_q29 = px.pie(
                q29_counts,
                names='Top-up Amount',
                values='Count',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_q29.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
            st.plotly_chart(fig_q29, use_container_width=True, key='topup_chart')

    # Third row: Types of Crops Grown (Co-occurrence Heatmap)
    st.subheader('Crop Co-occurrence Matrix')
    st.caption('Diagonal shows percentage of farmers growing each crop. Off-diagonal shows percentage of farmers growing both crops.')

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
    available_crops = {col: label for col, label in q06_labels.items() if col in df.columns}

    if len(available_crops) > 0:
        # Create co-occurrence matrix
        crop_cols = list(available_crops.keys())
        crop_names = list(available_crops.values())

        # Initialize matrix
        import numpy as np
        total_respondents = len(df)
        cooccurrence_matrix = np.zeros((len(crop_cols), len(crop_cols)))

        for i, crop1 in enumerate(crop_cols):
            for j, crop2 in enumerate(crop_cols):
                if i == j:
                    # Diagonal: percentage of farmers growing this crop
                    count = (df[crop1] == 1.0).sum()
                    cooccurrence_matrix[i, j] = (count / total_respondents * 100) if total_respondents > 0 else 0
                elif i > j:
                    # Lower diagonal: percentage of farmers growing both crops
                    count = ((df[crop1] == 1.0) & (df[crop2] == 1.0)).sum()
                    cooccurrence_matrix[i, j] = (count / total_respondents * 100) if total_respondents > 0 else 0
                else:
                    # Upper diagonal: set to NaN (will appear white)
                    cooccurrence_matrix[i, j] = np.nan

        # Create heatmap with RdBu colormap (Red to Blue)
        fig_crops = px.imshow(
            cooccurrence_matrix,
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
            height=600
        )
        fig_crops.update_xaxes(side='bottom', tickangle=-45)
        st.plotly_chart(fig_crops, use_container_width=True, key='crops_heatmap')


with tab2:
    st.header('Digital Behavior & Engagement')

    # First row: Phone sharing and Social Media Channels
    col1, col2 = st.columns(2)

    with col1:
        # Q07: Who do they share their phone with?
        st.subheader('Who do they share their phone with?')
        q07_labels = {
            'Q07_a': 'Children',
            'Q07_b': 'Spouse',
            'Q07_c': 'Cousin',
            'Q07_d': 'Parents',
            'Q07_e': 'Others',
            'Q07_f': "I don't share with anyone"
        }
        q07_data = []
        for col_name, label in q07_labels.items():
            if col_name in df.columns:
                count = (df[col_name] == 1.0).sum()
                q07_data.append({'Category': label, 'Count': count})

        q07_df = pd.DataFrame(q07_data)

        fig_q07 = px.pie(
            q07_df,
            names='Category',
            values='Count',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_q07.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
        st.plotly_chart(fig_q07, use_container_width=True, key='phone_sharing_pie')

    with col2:
        # Q10: Which social media channels are they using day to day? (UpSet plot)
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

        # Prepare data for UpSet plot
        q10_cols = [col for col in q10_labels.keys() if col in df.columns]

        # Create a DataFrame for upset plot - need to convert to boolean
        upset_q10_data = df[q10_cols].copy()
        upset_q10_data = upset_q10_data.fillna(0)
        upset_q10_data = (upset_q10_data == 1.0)

        # Rename columns to friendly names
        rename_q10_map = {col: q10_labels[col] for col in q10_cols if col in q10_labels}
        upset_q10_data = upset_q10_data.rename(columns=rename_q10_map)

        # Count combinations
        from collections import Counter
        q10_combinations = []
        for idx, row in upset_q10_data.iterrows():
            combo = tuple(col for col in upset_q10_data.columns if row[col])
            if combo:  # Only include if at least one channel is selected
                q10_combinations.append(combo)

        q10_combo_counts = Counter(q10_combinations)

        # Get top 15 combinations
        q10_top_combos = sorted(q10_combo_counts.items(), key=lambda x: x[1], reverse=True)[:15]

        # Create upset-style visualization manually
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Prepare data
        q10_combo_labels = []
        q10_combo_values = []
        q10_combo_percentages = []
        q10_matrix_data = {col: [] for col in upset_q10_data.columns}

        total_respondents = len(df)

        for combo, count in q10_top_combos:
            q10_combo_labels.append(' & '.join(combo) if len(combo) <= 2 else f"{len(combo)} channels")
            q10_combo_values.append(count)
            pct = (count / total_respondents * 100) if total_respondents > 0 else 0
            q10_combo_percentages.append(pct)

            for col in upset_q10_data.columns:
                q10_matrix_data[col].append(1 if col in combo else 0)

        # Create figure with subplots
        fig_q10 = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.1,
            specs=[[{"type": "bar"}], [{"type": "scatter"}]]
        )

        # Top plot: bar chart of combination percentages
        fig_q10.add_trace(
            go.Bar(
                x=list(range(len(q10_combo_percentages))),
                y=q10_combo_percentages,
                marker_color='#4575b4',
                showlegend=False,
                width=0.6,
                text=[f'{p:.1f}%' for p in q10_combo_percentages],
                textposition='outside',
                hovertemplate='%{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Bottom plot: matrix showing which channels are in each combination
        # Add connecting lines first
        for j in range(len(q10_matrix_data[list(upset_q10_data.columns)[0]])):
            # Find which channels are active for this combination
            active_indices = [i for i, col in enumerate(upset_q10_data.columns) if q10_matrix_data[col][j] == 1]

            if len(active_indices) > 1:
                # Draw connecting line
                fig_q10.add_trace(
                    go.Scatter(
                        x=[j] * len(active_indices),
                        y=active_indices,
                        mode='lines',
                        line=dict(color='#d73027', width=2),
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
                    marker=dict(size=10, color='#d73027', line=dict(color='#ffffff', width=1)),
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
            range=[-0.5, len(q10_combo_values) - 0.5],
            row=1, col=1
        )
        fig_q10.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, len(q10_combo_values) - 0.5],
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
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_q10, use_container_width=True, key='social_media_upset')

    # Second row: Time of Day and Hours per day
    col3, col4 = st.columns(2)

    with col3:
        # Q11: Time of Day Usage Pattern
        st.subheader('Internet Usage by Time of Day')

        q11_labels = {
            'Q11_a': 'Morning',
            'Q11_b': 'Afternoon',
            'Q11_c': 'Evening',
            'Q11_d': 'Night'
        }

        # Prepare data for time series line plot
        q11_data = []
        time_order = ['Morning', 'Afternoon', 'Evening', 'Night']

        for col_name, label in q11_labels.items():
            if col_name in df.columns:
                count = (df[col_name] == 1.0).sum()
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                q11_data.append({
                    'Time of Day': label,
                    'Count': count,
                    'Percentage': pct
                })

        q11_df = pd.DataFrame(q11_data)

        # Sort by time order
        q11_df['Time of Day'] = pd.Categorical(q11_df['Time of Day'], categories=time_order, ordered=True)
        q11_df = q11_df.sort_values('Time of Day')

        # Create vertical line plot with time on x-axis and percentage on y-axis
        import plotly.graph_objects as go

        fig_q11 = go.Figure()

        # Add line trace
        fig_q11.add_trace(go.Scatter(
            x=q11_df['Time of Day'],
            y=q11_df['Percentage'],
            mode='lines+markers+text',
            line=dict(color='#4575b4', width=3),
            marker=dict(
                size=12,
                color=q11_df['Percentage'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Usage %",
                    thickness=15,
                    len=0.7
                ),
                line=dict(color='#333', width=2)
            ),
            text=q11_df['Percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='top center',
            textfont=dict(size=12),
            hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
        ))

        fig_q11.update_layout(
            xaxis_title='Time of Day',
            yaxis_title='Percentage of Users (%)',
            height=400,
            yaxis=dict(
                range=[0, max(q11_df['Percentage']) * 1.15],
                showgrid=True,
                gridcolor='#e0e0e0'
            ),
            xaxis=dict(
                categoryorder='array',
                categoryarray=time_order
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.plotly_chart(fig_q11, use_container_width=True, key='time_of_day_line')

    with col4:
        # Q28: Hours per day
        st.subheader('Hours per day')
        if 'Q28' in df.columns:
            q28_counts = df['Q28'].value_counts().reset_index()
            q28_counts.columns = ['Hours', 'Count']

            fig_q28 = px.pie(
                q28_counts,
                names='Hours',
                values='Count',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_q28.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
            st.plotly_chart(fig_q28, use_container_width=True, key='hours_per_day_pie')

    # Third row: Level of poor connection and Where they heard about Yetagon
    st.subheader('Level of poor connection')

    if 'Q27' in df.columns:
        q27_counts = df['Q27'].value_counts().reset_index()
        q27_counts.columns = ['Connection Level', 'Count']

        # Capitalize first letter and add line breaks for better display
        q27_counts['Connection Level'] = q27_counts['Connection Level'].str.capitalize()

        # Split long labels intelligently at word boundaries
        def split_label(text, max_len=15):
            if len(text) <= max_len:
                return text

            words = text.split(' ')
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                # Check if adding this word exceeds the limit
                if current_length + len(word) + (1 if current_line else 0) > max_len:
                    if current_line:  # Save current line and start new one
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:  # Single word is too long, add it anyway
                        lines.append(word)
                        current_length = 0
                else:
                    current_line.append(word)
                    current_length += len(word) + (1 if len(current_line) > 1 else 0)

            if current_line:
                lines.append(' '.join(current_line))

            return '<br>'.join(lines)

        q27_counts['Connection Level'] = q27_counts['Connection Level'].apply(split_label)

        fig_q27 = px.pie(
            q27_counts,
            names='Connection Level',
            values='Count',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_q27.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
        st.plotly_chart(fig_q27, use_container_width=True, key='connection_level_pie')


with tab3:
    st.header('Internet Usage Behaviors')

    # Create sub-tabs for each platform
    platform_tabs = st.tabs(['Facebook', 'TikTok', 'Viber', 'YouTube'])

    # Facebook Tab
    with platform_tabs[0]:
        # Define confidence level mapping for Facebook
        fb_confidence_map = {
            'using news feed only': 'Newsfeed only',
            'Can use search': 'Can use search',
            'Can use Facebook actively (post, share, comment)': 'Engage Actively'
        }

        create_platform_analysis(
            df=df,
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
            df=df,
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
            df=df,
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
            df=df,
            platform_name='YouTube',
            platform_key='yt',
            platform_col='Q10_b',
            usecase_col='Q15',
            confidence_col='Q17',
            challenge_col='Q16'
        )

with tab4:
    st.header('Brand & Product Awareness')

    # Brand Awareness Section
    st.subheader('Brand Awareness')

    # Create filters for Brand Awareness
    brand_filter_col1, brand_filter_col2, brand_filter_col3, brand_filter_col4 = st.columns(4)

    with brand_filter_col1:
        digital_filter_brand = st.selectbox('Digital/Direct', ['ALL', 'DIGITAL', 'DIRECT'], key='brand_dd')

    with brand_filter_col2:
        gender_filter_brand = st.selectbox('Gender', ['All', 'Male', 'Female'], key='brand_gender')

    with brand_filter_col3:
        age_filter_brand = st.selectbox('Age Group', ['All', 'Under 20', '20-30', '31-40', '41-50', '51-60', 'Over 60'], key='brand_age')

    with brand_filter_col4:
        all_regions_brand = ['All'] + sorted(df['Region'].unique().tolist())
        region_filter_brand = st.selectbox('Region', all_regions_brand, key='brand_region')

    # Filter data based on selections for Brand Awareness
    brand_filtered = df.copy()

    if digital_filter_brand != 'ALL':
        brand_filtered = brand_filtered[brand_filtered['Direct or Digital?'] == digital_filter_brand.title()]

    if gender_filter_brand != 'All' and 'Q03' in brand_filtered.columns:
        brand_filtered = brand_filtered[brand_filtered['Q03'] == gender_filter_brand]

    if age_filter_brand != 'All' and 'Q02' in brand_filtered.columns:
        brand_filtered = brand_filtered[brand_filtered['Q02'] == age_filter_brand]

    if region_filter_brand != 'All':
        brand_filtered = brand_filtered[brand_filtered['Region'] == region_filter_brand]

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

    # Create filters for Product Awareness
    prod_filter_col1, prod_filter_col2, prod_filter_col3, prod_filter_col4 = st.columns(4)

    with prod_filter_col1:
        digital_filter_prod = st.selectbox('Digital/Direct', ['ALL', 'DIGITAL', 'DIRECT'], key='prod_dd')

    with prod_filter_col2:
        gender_filter_prod = st.selectbox('Gender', ['All', 'Male', 'Female'], key='prod_gender')

    with prod_filter_col3:
        age_filter_prod = st.selectbox('Age Group', ['All', 'Under 20', '20-30', '31-40', '41-50', '51-60', 'Over 60'], key='prod_age')

    with prod_filter_col4:
        all_regions_prod = ['All'] + sorted(df['Region'].unique().tolist())
        region_filter_prod = st.selectbox('Region', all_regions_prod, key='prod_region')

    # Filter data based on selections
    prod_filtered = df.copy()

    if digital_filter_prod != 'ALL':
        prod_filtered = prod_filtered[prod_filtered['Direct or Digital?'] == digital_filter_prod.title()]

    if gender_filter_prod != 'All' and 'Q03' in prod_filtered.columns:
        prod_filtered = prod_filtered[prod_filtered['Q03'] == gender_filter_prod]

    if age_filter_prod != 'All' and 'Q02' in prod_filtered.columns:
        prod_filtered = prod_filtered[prod_filtered['Q02'] == age_filter_prod]

    if region_filter_prod != 'All':
        prod_filtered = prod_filtered[prod_filtered['Region'] == region_filter_prod]

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
        'Q31_k': 'EM',
        'Q31_l': 'Other'
    }

    q33_labels = {
        'Q33_a': 'Yetagon Irrigation',
        'Q33_b': 'Yetagon EM/Zarmani',
        'Q33_c': 'Yetagon Trichoderma/Barhmati',
        'Q33_d': 'Yetagon Tele-agronomy',
        'Q33_i': 'Yetagon Sun-kissed',
        'Q33_j': 'Yetagon Fish Amino',
        'Q33_k': 'Po Chat',
        'Q33_l': 'Messenger',
        'Q33_f': 'Digital farm practices',
        'Q33_h': 'Other'
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
        'Q32_f': 'Gypsum Appication',
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
        'Q34_f': 'Gypsum Appication',
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
                    'Yetagon Sun-kissed': 'Q33_i',
                    'Yetagon Fish Amino': 'Q33_j',
                    'Po Chat': 'Q33_k',
                    'Messenger': 'Q33_l',
                    'Digital farm practices': 'Q33_f'
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
            'Yetagon Sun-kissed': 'Q33_i',
            'Yetagon Fish Amino': 'Q33_j',
            'Po Chat': 'Q33_k',
            'Messenger': 'Q33_l',
            'Digital farm practices': 'Q33_f'
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
            # Create a stacked bar chart
            st.markdown('#### NPS Distribution by Product')
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
    else:
        st.info('NPS data (Q40) not available in the dataset.')
