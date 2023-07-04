import pandas as pd

def break_into_categories(df):
    return df.groupby('Category')[['Category']].count()

def get_price_data(df):
    df_pivot = df.pivot_table(index=['Category','Type'], values='App', aggfunc='count')
    df_pivot = df_pivot.reset_index()
    return pd.DataFrame(df_pivot)

def merge_df(df, reviews):
    return pd.merge(df, reviews, on='App')

def get_sentiment_data(df, reviews):
    merged_df = merge_df(df, reviews)
    
    grouped_sentiment_category_count = merged_df.groupby(['Category', 'Sentiment']).agg({'App': 'count'}).reset_index()
    grouped_sentiment_category_sum = merged_df.groupby(['Category']).agg({'Sentiment': 'count'}).reset_index()

    new_df = pd.merge(grouped_sentiment_category_count, grouped_sentiment_category_sum, on=["Category"])
    new_df['Sentiment_Normalized'] = new_df.App/new_df.Sentiment_y
    new_df = new_df.groupby('Category').filter(lambda x: len(x) ==3)

    return new_df