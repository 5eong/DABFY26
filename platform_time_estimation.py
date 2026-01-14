"""
Generate platform usage statistics for dashboard visualization.
Run with: python platform_time_estimation.py
"""

import pandas as pd


def calculate_platform_usage_stats(df):
    """Calculate platform usage statistics"""
    platform_cols = {
        'Q10_a': 'Facebook',
        'Q10_b': 'Youtube',
        'Q10_c': 'TikTok',
        'Q10_d': 'Viber',
        'Q10_e': 'Instagram',
        'Q10_f': 'Messenger',
        'Q10_g': 'Others'
    }

    stats = []
    for col, platform in platform_cols.items():
        if col in df.columns:
            users = df[df[col] == 1.0]
            n_users = len(users)
            time_dist = users['Q28'].value_counts()

            # Get counts for each category
            lt2h = time_dist.get('Less than 2 hours', 0)
            t2_4h = time_dist.get('2-4 hours', 0)
            gt4h = time_dist.get('more than 4 hours', 0)

            # Normalize to 100% across only these three categories
            total = lt2h + t2_4h + gt4h
            if total > 0:
                pct_lt2h = lt2h / total * 100
                pct_2_4h = t2_4h / total * 100
                pct_gt4h = gt4h / total * 100
            else:
                pct_lt2h = pct_2_4h = pct_gt4h = 0

            stats.append({
                'Platform': platform,
                'N_Users': n_users,
                'Pct_of_Total': n_users / len(df) * 100,
                'Pct_LessThan2h': pct_lt2h,
                'Pct_2to4h': pct_2_4h,
                'Pct_MoreThan4h': pct_gt4h
            })

    return pd.DataFrame(stats)


def main():
    print("Generating platform usage statistics...")

    df = pd.read_csv('clean/CLEAN_FY26.csv')
    print(f"Total respondents: {len(df)}")

    usage_stats = calculate_platform_usage_stats(df)
    usage_stats.to_csv('clean/platform_usage_statistics.csv', index=False)

    print("Saved: clean/platform_usage_statistics.csv")
    print(usage_stats.to_string(index=False))


if __name__ == "__main__":
    main()
