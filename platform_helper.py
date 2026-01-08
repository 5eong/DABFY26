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


def apply_filters(df, filters):
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


def plot_demographics(df, platform_key):
    """
    Create demographic visualizations (4 plots in a row)

    Args:
        df: Filtered dataframe
        platform_key: Unique key prefix for charts
    """
    st.markdown('### Demographics')
    demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)

    with demo_col1:
        # Age Range
        st.markdown('#### Age Range')
        if 'Q02' in df.columns:
            age_counts = df['Q02'].value_counts().reset_index()
            age_counts.columns = ['Age', 'Count']
            age_order = ['Under 20', '20-30', '31-40', '41-50', '51-60', 'Over 60']
            age_counts['Age'] = pd.Categorical(age_counts['Age'], categories=age_order, ordered=True)
            age_counts = age_counts.sort_values('Age')

            fig_age = px.bar(
                age_counts,
                x='Age',
                y='Count',
                color_discrete_sequence=['#0f4c3a']
            )
            fig_age.update_layout(
                showlegend=False,
                xaxis_title='',
                yaxis_title='',
                height=250,
                annotations=[
                    dict(
                        text=f'n={len(df)}',
                        xref='paper', yref='paper',
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=10, color='gray'),
                        bgcolor='rgba(255,255,255,0.8)',
                        borderpad=4
                    )
                ]
            )
            fig_age.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_age, use_container_width=True, key=f'{platform_key}_age_chart')

    with demo_col2:
        # Direct vs Digital
        st.markdown('#### Direct vs Digital')
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
            fig_dd.update_layout(
                showlegend=False,
                height=250,
                annotations=[
                    dict(
                        text=f'n={len(df)}',
                        xref='paper', yref='paper',
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=10, color='gray'),
                        bgcolor='rgba(255,255,255,0.8)',
                        borderpad=4
                    )
                ]
            )
            st.plotly_chart(fig_dd, use_container_width=True, key=f'{platform_key}_dd_chart')

    with demo_col3:
        # Gender
        st.markdown('#### Gender')
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
            fig_gender.update_layout(
                showlegend=False,
                height=250,
                annotations=[
                    dict(
                        text=f'n={len(df)}',
                        xref='paper', yref='paper',
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=10, color='gray'),
                        bgcolor='rgba(255,255,255,0.8)',
                        borderpad=4
                    )
                ]
            )
            st.plotly_chart(fig_gender, use_container_width=True, key=f'{platform_key}_gender_chart')

    with demo_col4:
        # Region
        st.markdown('#### Region')
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
            fig_region.update_layout(
                showlegend=False,
                height=250,
                annotations=[
                    dict(
                        text=f'n={len(df)}',
                        xref='paper', yref='paper',
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=10, color='gray'),
                        bgcolor='rgba(255,255,255,0.8)',
                        borderpad=4
                    )
                ]
            )
            st.plotly_chart(fig_region, use_container_width=True, key=f'{platform_key}_region_chart')


def plot_usage_habits(df, platform_key, usecase_col_prefix, confidence_col, confidence_map=None):
    """
    Create usage habit visualizations (2 plots in a row)

    Args:
        df: Filtered dataframe
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
        usecase_data = []
        for col_name, label in usecase_labels.items():
            if col_name in df.columns:
                count = (df[col_name] == 1.0).sum()
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                usecase_data.append({'Use Case': label, 'Percentage': pct})

        usecase_df = pd.DataFrame(usecase_data)
        usecase_df = usecase_df.sort_values('Percentage', ascending=True)

        fig_usecase = px.bar(
            usecase_df,
            y='Use Case',
            x='Percentage',
            orientation='h',
            text='Percentage',
            color_discrete_sequence=['#5a8f7b']
        )
        fig_usecase.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_usecase.update_layout(
            showlegend=False,
            xaxis_title='',
            yaxis_title='',
            height=300,
            annotations=[
                dict(
                    text=f'n={len(df)}',
                    xref='paper', yref='paper',
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10, color='gray'),
                    bgcolor='rgba(255,255,255,0.8)',
                    borderpad=4
                )
            ]
        )
        st.plotly_chart(fig_usecase, use_container_width=True, key=f'{platform_key}_usecase_chart')

    with usage_col2:
        # Confidence Level
        st.markdown('#### Confidence Level')
        if confidence_col in df.columns:
            conf_counts = df[confidence_col].value_counts().reset_index()
            conf_counts.columns = ['Level', 'Count']
            conf_counts['Percentage'] = (conf_counts['Count'] / conf_counts['Count'].sum() * 100).round(1)

            if confidence_map:
                conf_counts['Level'] = conf_counts['Level'].map(confidence_map).fillna(conf_counts['Level'])

            conf_counts = conf_counts.sort_values('Percentage', ascending=True)

            fig_conf = px.bar(
                conf_counts,
                y='Level',
                x='Percentage',
                orientation='h',
                text='Percentage',
                color_discrete_sequence=['#5a8f7b']
            )
            fig_conf.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_conf.update_layout(
                showlegend=False,
                xaxis_title='',
                yaxis_title='',
                height=300,
                annotations=[
                    dict(
                        text=f'n={len(df)}',
                        xref='paper', yref='paper',
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=10, color='gray'),
                        bgcolor='rgba(255,255,255,0.8)',
                        borderpad=4
                    )
                ]
            )
            st.plotly_chart(fig_conf, use_container_width=True, key=f'{platform_key}_conf_chart')


def plot_challenges(df, platform_key, challenge_col_prefix):
    """
    Create challenge visualizations (2 plots in a row)

    Args:
        df: Filtered dataframe
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
        challenge_data = []
        for col_name, label in challenge_labels.items():
            if col_name in df.columns:
                count = (df[col_name] == 1.0).sum()
                pct = (count / len(df) * 100) if len(df) > 0 else 0
                challenge_data.append({'Challenge': label, 'Percentage': pct})

        challenge_df = pd.DataFrame(challenge_data)
        challenge_df = challenge_df.sort_values('Percentage', ascending=True)

        fig_challenges = px.bar(
            challenge_df,
            y='Challenge',
            x='Percentage',
            orientation='h',
            text='Percentage',
            color_discrete_sequence=['#d73027']
        )
        fig_challenges.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_challenges.update_layout(
            showlegend=False,
            xaxis_title='',
            yaxis_title='',
            height=300,
            annotations=[
                dict(
                    text=f'n={len(df)}',
                    xref='paper', yref='paper',
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10, color='gray'),
                    bgcolor='rgba(255,255,255,0.8)',
                    borderpad=4
                )
            ]
        )
        st.plotly_chart(fig_challenges, use_container_width=True, key=f'{platform_key}_challenges_chart')

    with challenge_col2:
        # Level of poor connection
        st.markdown('#### Level of Poor Connection')
        if 'Q27' in df.columns:
            conn_counts = df['Q27'].value_counts().reset_index()
            conn_counts.columns = ['Level', 'Count']
            conn_counts['Percentage'] = (conn_counts['Count'] / conn_counts['Count'].sum() * 100).round(1)
            conn_counts['Level'] = conn_counts['Level'].str.capitalize()
            conn_counts = conn_counts.sort_values('Percentage', ascending=True)

            fig_conn = px.bar(
                conn_counts,
                y='Level',
                x='Percentage',
                orientation='h',
                text='Percentage',
                color_discrete_sequence=['#d73027']
            )
            fig_conn.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_conn.update_layout(
                showlegend=False,
                xaxis_title='',
                yaxis_title='',
                height=300,
                annotations=[
                    dict(
                        text=f'n={len(df)}',
                        xref='paper', yref='paper',
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=10, color='gray'),
                        bgcolor='rgba(255,255,255,0.8)',
                        borderpad=4
                    )
                ]
            )
            st.plotly_chart(fig_conn, use_container_width=True, key=f'{platform_key}_conn_chart')


def create_platform_analysis(df, platform_name, platform_key, platform_col, usecase_col, confidence_col, challenge_col, confidence_map=None):
    """
    Create complete platform analysis with filters and all visualizations

    Args:
        df: Full dataframe
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

    # Filter data - only users who use this platform
    filtered = df[df[platform_col] == 1.0].copy()
    filtered = apply_filters(filtered, filters)

    # Create visualizations
    plot_demographics(filtered, platform_key)
    plot_usage_habits(filtered, platform_key, usecase_col, confidence_col, confidence_map)
    plot_challenges(filtered, platform_key, challenge_col)
