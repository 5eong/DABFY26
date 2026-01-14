"""
Helper functions for platform-specific analysis in the Internet Usage Behaviors tab
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_platform_filters(platform_key, df):
    """
    Create filter dropdowns for a platform

    Args:
        platform_key: Unique key prefix for the platform (e.g., 'fb', 'tt')
        df: Full dataframe

    Returns:
        Dictionary with filter values
    """
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        dd_filter = st.selectbox('Digital/Direct', ['ALL', 'DIGITAL', 'DIRECT'], key=f'{platform_key}_dd')

    with filter_col2:
        gender_filter = st.selectbox('Gender', ['ALL', 'Male', 'Female'], key=f'{platform_key}_gender')

    with filter_col3:
        age_options = ['ALL'] + ['Under 20', '20-30', '31-40', '41-50', '51-60', 'Over 60']
        age_filter = st.selectbox('Age', age_options, key=f'{platform_key}_age')

    with filter_col4:
        hours_options = ['ALL'] + sorted(df['Q28'].dropna().unique().tolist()) if 'Q28' in df.columns else ['ALL']
        hours_filter = st.selectbox('Hours Per Day', hours_options, key=f'{platform_key}_hours')

    return {
        'dd': dd_filter,
        'gender': gender_filter,
        'age': age_filter,
        'hours': hours_filter
    }


def apply_platform_filters(df, filters):
    """
    Apply filters to dataframe

    Args:
        df: Filtered dataframe (already filtered by platform usage)
        filters: Dictionary with filter values from create_platform_filters

    Returns:
        Filtered dataframe
    """
    filtered = df.copy()

    if filters['dd'] != 'ALL':
        if 'Direct or Digital?' in filtered.columns:
            filtered = filtered[filtered['Direct or Digital?'] == filters['dd'].title()]

    if filters['gender'] != 'ALL':
        if 'Q03' in filtered.columns:
            filtered = filtered[filtered['Q03'] == filters['gender']]

    if filters['age'] != 'ALL':
        if 'Q02' in filtered.columns:
            filtered = filtered[filtered['Q02'] == filters['age']]

    if filters['hours'] != 'ALL':
        if 'Q28' in filtered.columns:
            filtered = filtered[filtered['Q28'] == filters['hours']]

    return filtered


def plot_demographics(df, df_fy25, platform_key):
    """
    Create demographic visualizations with FY25 comparison
    Row 1: Age Range, Direct vs Digital (with FY25 on left)
    Row 2: Gender, Region (with FY25 on left)

    Args:
        df: Filtered FY26 dataframe
        df_fy25: Filtered FY25 dataframe
        platform_key: Unique key prefix for charts
    """
    st.markdown('### Demographics')

    # Row 1: Age Range and Direct vs Digital
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        # Age Range - Bar with FY25 line overlay
        st.markdown('#### Age Range')
        age_order = ['Under 20', '20-30', '31-40', '41-50', '51-60', 'Over 60']

        if 'Q02' in df.columns:
            # FY26 data
            age_counts_fy26 = df['Q02'].value_counts().reset_index()
            age_counts_fy26.columns = ['Age', 'Count']
            age_counts_fy26['Age'] = pd.Categorical(age_counts_fy26['Age'], categories=age_order, ordered=True)
            age_counts_fy26 = age_counts_fy26.sort_values('Age')
            total_fy26 = age_counts_fy26['Count'].sum()
            age_counts_fy26['Percentage'] = (age_counts_fy26['Count'] / total_fy26 * 100) if total_fy26 > 0 else 0

            # FY25 data
            age_counts_fy25 = None
            if df_fy25 is not None and 'Q02' in df_fy25.columns and len(df_fy25) > 0:
                age_counts_fy25 = df_fy25['Q02'].value_counts().reset_index()
                age_counts_fy25.columns = ['Age', 'Count']
                age_counts_fy25['Age'] = pd.Categorical(age_counts_fy25['Age'], categories=age_order, ordered=True)
                age_counts_fy25 = age_counts_fy25.sort_values('Age')
                total_fy25 = age_counts_fy25['Count'].sum()
                age_counts_fy25['Percentage'] = (age_counts_fy25['Count'] / total_fy25 * 100) if total_fy25 > 0 else 0

            fig_age = go.Figure()

            # FY26 bars
            fig_age.add_trace(go.Bar(
                x=age_counts_fy26['Age'],
                y=age_counts_fy26['Percentage'],
                name='FY26',
                marker_color='#4575b4',
                text=[f'{p:.1f}%' for p in age_counts_fy26['Percentage']],
                textposition='outside',
                hovertemplate='FY26<br>%{x}<br>%{y:.1f}%<extra></extra>'
            ))

            # FY25 line
            if age_counts_fy25 is not None:
                fig_age.add_trace(go.Scatter(
                    x=age_counts_fy25['Age'],
                    y=age_counts_fy25['Percentage'],
                    name='FY25',
                    mode='lines+markers',
                    line=dict(color='#d73027', width=2, dash='dash'),
                    marker=dict(size=8, color='#d73027'),
                    hovertemplate='FY25<br>%{x}<br>%{y:.1f}%<extra></extra>'
                ))

            fig_age.update_layout(
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis_title='',
                yaxis_title='%',
                height=280,
                xaxis=dict(categoryorder='array', categoryarray=age_order)
            )
            fig_age.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_age, use_container_width=True, key=f'{platform_key}_age_chart')

    with row1_col2:
        # Direct vs Digital - Side by side pies
        st.markdown('#### Direct vs Digital')
        dd_col1, dd_col2 = st.columns(2)

        with dd_col1:
            st.markdown('**FY25**')
            if df_fy25 is not None and 'Direct or Digital?' in df_fy25.columns and len(df_fy25) > 0:
                dd_counts_fy25 = df_fy25['Direct or Digital?'].value_counts().reset_index()
                dd_counts_fy25.columns = ['Type', 'Count']

                fig_dd_fy25 = px.pie(
                    dd_counts_fy25,
                    names='Type',
                    values='Count',
                    hole=0.4,
                    color='Type',
                    color_discrete_map={'Digital': '#0f4c3a', 'Direct': '#8fc1e3'}
                )
                fig_dd_fy25.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_dd_fy25.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_dd_fy25, use_container_width=True, key=f'{platform_key}_dd_chart_fy25')

        with dd_col2:
            st.markdown('**FY26**')
            if 'Direct or Digital?' in df.columns:
                dd_counts = df['Direct or Digital?'].value_counts().reset_index()
                dd_counts.columns = ['Type', 'Count']

                fig_dd = px.pie(
                    dd_counts,
                    names='Type',
                    values='Count',
                    hole=0.4,
                    color='Type',
                    color_discrete_map={'Digital': '#0f4c3a', 'Direct': '#8fc1e3'}
                )
                fig_dd.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_dd.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_dd, use_container_width=True, key=f'{platform_key}_dd_chart')

    # Row 2: Gender and Region
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        # Gender - Side by side pies
        st.markdown('#### Gender')
        gender_col1, gender_col2 = st.columns(2)

        with gender_col1:
            st.markdown('**FY25**')
            if df_fy25 is not None and 'Q03' in df_fy25.columns and len(df_fy25) > 0:
                gender_counts_fy25 = df_fy25['Q03'].value_counts().reset_index()
                gender_counts_fy25.columns = ['Gender', 'Count']

                fig_gender_fy25 = px.pie(
                    gender_counts_fy25,
                    names='Gender',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_gender_fy25.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_gender_fy25.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_gender_fy25, use_container_width=True, key=f'{platform_key}_gender_chart_fy25')

        with gender_col2:
            st.markdown('**FY26**')
            if 'Q03' in df.columns:
                gender_counts = df['Q03'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']

                fig_gender = px.pie(
                    gender_counts,
                    names='Gender',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_gender.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_gender.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_gender, use_container_width=True, key=f'{platform_key}_gender_chart')

    with row2_col2:
        # Region - Side by side pies
        st.markdown('#### Region')
        region_col1, region_col2 = st.columns(2)

        with region_col1:
            st.markdown('**FY25**')
            # FY25 uses Q04 for region
            fy25_region_col = 'Q04' if df_fy25 is not None and 'Q04' in df_fy25.columns else 'Region'
            if df_fy25 is not None and fy25_region_col in df_fy25.columns and len(df_fy25) > 0:
                region_counts_fy25 = df_fy25[fy25_region_col].value_counts().reset_index()
                region_counts_fy25.columns = ['Region', 'Count']

                fig_region_fy25 = px.pie(
                    region_counts_fy25,
                    names='Region',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_region_fy25.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_region_fy25.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_region_fy25, use_container_width=True, key=f'{platform_key}_region_chart_fy25')

        with region_col2:
            st.markdown('**FY26**')
            if 'Region' in df.columns:
                region_counts = df['Region'].value_counts().reset_index()
                region_counts.columns = ['Region', 'Count']

                fig_region = px.pie(
                    region_counts,
                    names='Region',
                    values='Count',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_region.update_traces(texttemplate='%{label}<br>%{percent:.1%}', textposition='auto')
                fig_region.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_region, use_container_width=True, key=f'{platform_key}_region_chart')


def plot_usage_habits(df, df_fy25, platform_key, usecase_col_prefix, confidence_col, confidence_map=None):
    """
    Create usage habit visualizations with FY25 comparison

    Args:
        df: Filtered FY26 dataframe
        df_fy25: Filtered FY25 dataframe
        platform_key: Unique key prefix for charts
        usecase_col_prefix: Question prefix for use cases (e.g., 'Q12' for Facebook)
        confidence_col: Question column for confidence level (e.g., 'Q14')
        confidence_map: Optional mapping for confidence levels
    """
    st.markdown('### Usage Habits')
    usage_col1, usage_col2 = st.columns(2)

    with usage_col1:
        # What do they use it for?
        st.markdown('#### What do they use it for?')
        usecase_labels = {
            f'{usecase_col_prefix}_a': 'To read the news',
            f'{usecase_col_prefix}_b': 'To find information about farming',
            f'{usecase_col_prefix}_c': 'To contact with friends and family',
            f'{usecase_col_prefix}_d': 'For entertainment',
            f'{usecase_col_prefix}_e': 'Other'
        }

        # FY26 data
        usecase_data_fy26 = []
        for col_name, label in usecase_labels.items():
            if col_name in df.columns:
                count = (df[col_name] == 1.0).sum()
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                usecase_data_fy26.append({'Use Case': label, 'Percentage': pct})

        usecase_df_fy26 = pd.DataFrame(usecase_data_fy26)
        usecase_df_fy26 = usecase_df_fy26.sort_values('Percentage', ascending=True)

        # FY25 data
        usecase_data_fy25 = []
        if df_fy25 is not None and len(df_fy25) > 0:
            for col_name, label in usecase_labels.items():
                if col_name in df_fy25.columns:
                    count = (df_fy25[col_name] == 1.0).sum()
                    pct = (count / len(df_fy25) * 100) if len(df_fy25) > 0 else 0
                    usecase_data_fy25.append({'Use Case': label, 'Percentage': pct})

        usecase_df_fy25 = pd.DataFrame(usecase_data_fy25) if usecase_data_fy25 else None
        if usecase_df_fy25 is not None and len(usecase_df_fy25) > 0:
            # Reorder to match FY26 order
            usecase_df_fy25 = usecase_df_fy25.set_index('Use Case').reindex(usecase_df_fy26['Use Case'].values).reset_index()

        fig_usecase = go.Figure()

        # FY26 bars
        fig_usecase.add_trace(go.Bar(
            y=usecase_df_fy26['Use Case'],
            x=usecase_df_fy26['Percentage'],
            name='FY26',
            orientation='h',
            marker_color='#4575b4',
            text=[f'{p:.1f}%' for p in usecase_df_fy26['Percentage']],
            textposition='outside',
            hovertemplate='FY26<br>%{y}<br>%{x:.1f}%<extra></extra>'
        ))

        # FY25 line
        if usecase_df_fy25 is not None and len(usecase_df_fy25) > 0:
            fig_usecase.add_trace(go.Scatter(
                y=usecase_df_fy25['Use Case'],
                x=usecase_df_fy25['Percentage'],
                name='FY25',
                mode='lines+markers',
                line=dict(color='#d73027', width=2, dash='dash'),
                marker=dict(size=8, color='#d73027'),
                hovertemplate='FY25<br>%{y}<br>%{x:.1f}%<extra></extra>'
            ))

        fig_usecase.update_layout(
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_title='%',
            yaxis_title='',
            height=300
        )
        st.plotly_chart(fig_usecase, use_container_width=True, key=f'{platform_key}_usecase_chart')

    with usage_col2:
        # Confidence Level
        st.markdown('#### Confidence Level')
        if confidence_col in df.columns:
            # FY26 data
            conf_counts_fy26 = df[confidence_col].value_counts().reset_index()
            conf_counts_fy26.columns = ['Level', 'Count']
            conf_counts_fy26['Percentage'] = (conf_counts_fy26['Count'] / conf_counts_fy26['Count'].sum() * 100).round(1)

            if confidence_map:
                conf_counts_fy26['Level'] = conf_counts_fy26['Level'].map(confidence_map).fillna(conf_counts_fy26['Level'])

            conf_counts_fy26 = conf_counts_fy26.sort_values('Percentage', ascending=True)

            # FY25 data
            conf_counts_fy25 = None
            if df_fy25 is not None and confidence_col in df_fy25.columns and len(df_fy25) > 0:
                conf_counts_fy25 = df_fy25[confidence_col].value_counts().reset_index()
                conf_counts_fy25.columns = ['Level', 'Count']
                conf_counts_fy25['Percentage'] = (conf_counts_fy25['Count'] / conf_counts_fy25['Count'].sum() * 100).round(1)

                if confidence_map:
                    conf_counts_fy25['Level'] = conf_counts_fy25['Level'].map(confidence_map).fillna(conf_counts_fy25['Level'])

                # Reorder to match FY26 order
                conf_counts_fy25 = conf_counts_fy25.set_index('Level').reindex(conf_counts_fy26['Level'].values).reset_index()

            fig_conf = go.Figure()

            # FY26 bars
            fig_conf.add_trace(go.Bar(
                y=conf_counts_fy26['Level'],
                x=conf_counts_fy26['Percentage'],
                name='FY26',
                orientation='h',
                marker_color='#4575b4',
                text=[f'{p:.1f}%' for p in conf_counts_fy26['Percentage']],
                textposition='outside',
                hovertemplate='FY26<br>%{y}<br>%{x:.1f}%<extra></extra>'
            ))

            # FY25 line
            if conf_counts_fy25 is not None and len(conf_counts_fy25) > 0:
                fig_conf.add_trace(go.Scatter(
                    y=conf_counts_fy25['Level'],
                    x=conf_counts_fy25['Percentage'],
                    name='FY25',
                    mode='lines+markers',
                    line=dict(color='#d73027', width=2, dash='dash'),
                    marker=dict(size=8, color='#d73027'),
                    hovertemplate='FY25<br>%{y}<br>%{x:.1f}%<extra></extra>'
                ))

            fig_conf.update_layout(
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis_title='%',
                yaxis_title='',
                height=300
            )
            st.plotly_chart(fig_conf, use_container_width=True, key=f'{platform_key}_conf_chart')


def plot_challenges(df, df_fy25, platform_key, challenge_col_prefix):
    """
    Create challenge visualizations with FY25 comparison

    Args:
        df: Filtered FY26 dataframe
        df_fy25: Filtered FY25 dataframe
        platform_key: Unique key prefix for charts
        challenge_col_prefix: Question prefix for challenges (e.g., 'Q13' for Facebook)
    """
    st.markdown('### Challenges')
    challenge_col1, challenge_col2 = st.columns(2)

    with challenge_col1:
        # Key challenges
        st.markdown('#### Key Challenges')
        challenge_labels = {
            f'{challenge_col_prefix}_a': 'Too complicated to use',
            f'{challenge_col_prefix}_b': 'Poor Internet',
            f'{challenge_col_prefix}_c': 'Internet price too expensive',
            f'{challenge_col_prefix}_d': 'Have to change mobile numbers',
            f'{challenge_col_prefix}_e': 'Other'
        }

        # FY26 data
        challenge_data_fy26 = []
        for col_name, label in challenge_labels.items():
            if col_name in df.columns:
                count = (df[col_name] == 1.0).sum()
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                challenge_data_fy26.append({'Challenge': label, 'Percentage': pct})

        challenge_df_fy26 = pd.DataFrame(challenge_data_fy26)
        challenge_df_fy26 = challenge_df_fy26.sort_values('Percentage', ascending=True)

        # FY25 data
        challenge_data_fy25 = []
        if df_fy25 is not None and len(df_fy25) > 0:
            for col_name, label in challenge_labels.items():
                if col_name in df_fy25.columns:
                    count = (df_fy25[col_name] == 1.0).sum()
                    pct = (count / len(df_fy25) * 100) if len(df_fy25) > 0 else 0
                    challenge_data_fy25.append({'Challenge': label, 'Percentage': pct})

        challenge_df_fy25 = pd.DataFrame(challenge_data_fy25) if challenge_data_fy25 else None
        if challenge_df_fy25 is not None and len(challenge_df_fy25) > 0:
            challenge_df_fy25 = challenge_df_fy25.set_index('Challenge').reindex(challenge_df_fy26['Challenge'].values).reset_index()

        fig_challenges = go.Figure()

        # FY26 bars
        fig_challenges.add_trace(go.Bar(
            y=challenge_df_fy26['Challenge'],
            x=challenge_df_fy26['Percentage'],
            name='FY26',
            orientation='h',
            marker_color='#4575b4',
            text=[f'{p:.1f}%' for p in challenge_df_fy26['Percentage']],
            textposition='outside',
            hovertemplate='FY26<br>%{y}<br>%{x:.1f}%<extra></extra>'
        ))

        # FY25 line
        if challenge_df_fy25 is not None and len(challenge_df_fy25) > 0:
            fig_challenges.add_trace(go.Scatter(
                y=challenge_df_fy25['Challenge'],
                x=challenge_df_fy25['Percentage'],
                name='FY25',
                mode='lines+markers',
                line=dict(color='#d73027', width=2, dash='dash'),
                marker=dict(size=8, color='#d73027'),
                hovertemplate='FY25<br>%{y}<br>%{x:.1f}%<extra></extra>'
            ))

        fig_challenges.update_layout(
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_title='%',
            yaxis_title='',
            height=300
        )
        st.plotly_chart(fig_challenges, use_container_width=True, key=f'{platform_key}_challenges_chart')

    with challenge_col2:
        # Level of poor connection
        st.markdown('#### Level of Poor Connection')
        if 'Q27' in df.columns:
            # FY26 data
            conn_counts_fy26 = df['Q27'].value_counts().reset_index()
            conn_counts_fy26.columns = ['Level', 'Count']
            conn_counts_fy26['Percentage'] = (conn_counts_fy26['Count'] / conn_counts_fy26['Count'].sum() * 100).round(1)
            conn_counts_fy26['Level'] = conn_counts_fy26['Level'].str.capitalize()
            conn_counts_fy26 = conn_counts_fy26.sort_values('Percentage', ascending=True)

            # FY25 data
            conn_counts_fy25 = None
            if df_fy25 is not None and 'Q27' in df_fy25.columns and len(df_fy25) > 0:
                conn_counts_fy25 = df_fy25['Q27'].value_counts().reset_index()
                conn_counts_fy25.columns = ['Level', 'Count']
                conn_counts_fy25['Percentage'] = (conn_counts_fy25['Count'] / conn_counts_fy25['Count'].sum() * 100).round(1)
                conn_counts_fy25['Level'] = conn_counts_fy25['Level'].str.capitalize()
                conn_counts_fy25 = conn_counts_fy25.set_index('Level').reindex(conn_counts_fy26['Level'].values).reset_index()

            fig_conn = go.Figure()

            # FY26 bars
            fig_conn.add_trace(go.Bar(
                y=conn_counts_fy26['Level'],
                x=conn_counts_fy26['Percentage'],
                name='FY26',
                orientation='h',
                marker_color='#4575b4',
                text=[f'{p:.1f}%' for p in conn_counts_fy26['Percentage']],
                textposition='outside',
                hovertemplate='FY26<br>%{y}<br>%{x:.1f}%<extra></extra>'
            ))

            # FY25 line
            if conn_counts_fy25 is not None and len(conn_counts_fy25) > 0:
                fig_conn.add_trace(go.Scatter(
                    y=conn_counts_fy25['Level'],
                    x=conn_counts_fy25['Percentage'],
                    name='FY25',
                    mode='lines+markers',
                    line=dict(color='#d73027', width=2, dash='dash'),
                    marker=dict(size=8, color='#d73027'),
                    hovertemplate='FY25<br>%{y}<br>%{x:.1f}%<extra></extra>'
                ))

            fig_conn.update_layout(
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                xaxis_title='%',
                yaxis_title='',
                height=300
            )
            st.plotly_chart(fig_conn, use_container_width=True, key=f'{platform_key}_conn_chart')


def create_platform_analysis(df, df_fy25, platform_name, platform_key, platform_col, usecase_col, confidence_col, challenge_col, confidence_map=None):
    """
    Create complete platform analysis with filters and all visualizations

    Args:
        df: Full FY26 dataframe
        df_fy25: Full FY25 dataframe
        platform_name: Display name (e.g., 'Facebook')
        platform_key: Unique key prefix (e.g., 'fb')
        platform_col: Column indicating platform usage (e.g., 'Q10_a')
        usecase_col: Question prefix for use cases (e.g., 'Q12')
        confidence_col: Question column for confidence (e.g., 'Q14')
        challenge_col: Question prefix for challenges (e.g., 'Q13')
        confidence_map: Optional mapping for confidence levels
    """
    st.subheader(f'{platform_name} Usage Analysis')

    # Create filters
    filters = create_platform_filters(platform_key, df)

    # Filter FY26 data - only users who use this platform
    filtered = df[df[platform_col] == 1.0].copy() if platform_col in df.columns else df.copy()
    filtered = apply_platform_filters(filtered, filters)

    # Filter FY25 data - only users who use this platform
    filtered_fy25 = None
    if df_fy25 is not None and platform_col in df_fy25.columns:
        filtered_fy25 = df_fy25[df_fy25[platform_col] == 1.0].copy()
        filtered_fy25 = apply_platform_filters(filtered_fy25, filters)

    # Display sample size as subtle caption
    fy25_n = len(filtered_fy25) if filtered_fy25 is not None else 0
    st.caption(f"n = {len(filtered):,} (FY26) · {fy25_n:,} (FY25)")

    # Create visualizations
    plot_demographics(filtered, filtered_fy25, platform_key)
    plot_usage_habits(filtered, filtered_fy25, platform_key, usecase_col, confidence_col, confidence_map)
    plot_challenges(filtered, filtered_fy25, platform_key, challenge_col)
