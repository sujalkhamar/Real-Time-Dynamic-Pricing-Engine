import pandas as pd

def get_season(month):
    peak = ['July', 'August', 'December']
    if month in peak:
        return 0.3
    else:
        return 0.1

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Drop column with too many missing values
    if 'company' in df.columns:
        df = df.drop(columns=['company'])

    # Handle missing values
    df['agent'] = df['agent'].fillna(-1)
    df['country'] = df['country'].fillna("Unknown")
    df['children'] = df['children'].fillna(0)

    # Feature Engineering
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['stay_length'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['season_factor'] = df['arrival_date_month'].apply(get_season)

    return df