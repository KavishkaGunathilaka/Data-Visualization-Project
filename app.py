import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime
import plotly.express as px
from tab1 import get_corelations, get_importances

st.set_page_config(
    page_title="Data Visualization - Group 8",
    page_icon="🧊",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv('data/googleplaystore.csv')
    df = df.loc[df.Rating.notna()].reset_index(drop=True)
    return df

@st.cache_data
def preprocess_data(df):
    enc_df = df.copy()

    le = preprocessing.LabelEncoder()
    enc_df['Category'] = le.fit_transform(df.Category)

    enc_df['Size'] = df.Size.str.strip('M').str.strip('k')
    enc_df.loc[enc_df.Size == 'Varies with device', 'Size'] = enc_df.loc[enc_df.Size != 'Varies with device'].Size.astype(float).mean()
    enc_df['Size'] = enc_df['Size'].astype(float)

    enc_df['Installs'] = df.Installs.str.strip('+').str.replace(',', '').astype(int)

    enc_df['Type'] = le.fit_transform(df.Type)

    enc_df['Price'] = df.Price.str.strip('$').astype(float)

    enc_df['Content Rating'] = le.fit_transform(df['Content Rating'])

    enc_df['Genres'] = le.fit_transform(df.Genres)

    def DateDiff(date1, date2):
        return (date2 - date1).days

    enc_df['Last Updated'] = df['Last Updated'].apply(pd.to_datetime).map(lambda x: DateDiff(x, datetime.datetime.strptime('29062023', "%d%m%Y")))

    enc_df = enc_df.drop(['App', 'Current Ver', 'Android Ver'], axis=1)

    return enc_df

def main():
    tab1, tab2 = st.tabs(["Question 1", "Question 2"])
    df = load_data()
    enc_df = preprocess_data(df)

    with tab1:
        st.header("How to increase ratings on the Play Store?")
        col1, col2 = st.columns(2)

        with st.container():
            with col1:
                corr = get_corelations(enc_df)
                fig = px.imshow(corr, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                feature_importances = get_importances(enc_df)
                fig = px.bar(feature_importances)
                st.plotly_chart(fig, use_container_width=True)

        with st.container():
            features = ('Category', 'Reviews', 'Installs', 'Price', 'Last Updated', 'Size', 'Content Rating')
            option = st.selectbox('Feature', features)

            if option == 'Category':
                fig = px.box(df, x="Category", y="Rating", color="Category", title="Rating of application in each category is not different too much")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Reviews':
                fig = px.scatter(df.loc[df.Reviews < 10000000], x="Reviews", y="Rating", trendline="ols",  trendline_color_override="red", title="As the number of reviews increase, the rating also increases")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Installs':
                sorted_value = sorted(list(enc_df['Installs'].unique()))
                temp = enc_df.copy()
                temp['Installs'].replace(sorted_value,range(0,len(sorted_value),1), inplace = True ) #label encide installs
                fig = px.scatter(temp, x="Installs", y="Rating", trendline="ols",  trendline_color_override="red", title="Seem like number of install affect to rating")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Price':
                fig = px.scatter(enc_df, x="Price", y="Rating", trendline="ols",  trendline_color_override="red", title="Higher price application may make customer disappointed, if they are not good enough")
                st.plotly_chart(fig, use_container_width=True)

                temp = enc_df.copy()
                temp.loc[ temp['Price'] == 0, 'PriceBand'] = 'Free'
                temp.loc[(temp['Price'] > 0) & (temp['Price'] <= 0.99), 'PriceBand'] = 'Cheap'
                temp.loc[(temp['Price'] > 0.99) & (temp['Price'] <= 2.99), 'PriceBand']   = 'Not cheap'
                temp.loc[(temp['Price'] > 2.99) & (temp['Price'] <= 4.99), 'PriceBand']   = 'Normal'
                temp.loc[(temp['Price'] > 4.99) & (temp['Price'] <= 14.99), 'PriceBand']   = 'Expensive'
                temp.loc[(temp['Price'] > 14.99) & (temp['Price'] <= 29.99), 'PriceBand']   = 'Too expensive'
                temp.loc[(temp['Price'] > 29.99), 'PriceBand']  = 'Very Very expensive'
                temp[['PriceBand', 'Rating']].groupby(['PriceBand'], as_index=False).mean()

                fig = px.box(temp, x="PriceBand", y="Rating", color="PriceBand", title="Price are not effect to rating ,but if it is very expensive, it might get low rating")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Last Updated':
                fig = px.scatter(enc_df, x="Last Updated", y="Rating", trendline="ols",  trendline_color_override="red", title="It seems like ratings of the apps are high when its regualry updated")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Size':
                fig = px.scatter(enc_df, x="Size", y="Rating", trendline="ols",  trendline_color_override="red", title="")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Content Rating':
                fig = px.box(df[df['Content Rating'] != 'Unrated'], x="Content Rating", y="Rating", color="Content Rating", title="Content Rating not effect too much to rating, but in Mature applications ,look like they get lower rating than other.")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Are there any market opportunities for new apps?")

if __name__ == '__main__':
    main()
    