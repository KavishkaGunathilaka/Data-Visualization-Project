import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tab1 import get_corelations, get_importances
from tab2 import break_into_categories, get_price_data, get_sentiment_data, merge_df

nltk.download('stopwords')

st.set_page_config(
    page_title="Data Visualization - Group 8",
    page_icon="📈",
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
    enc_df.loc[enc_df.Size == 'Varies with device', 'Size'] = enc_df.loc[enc_df.Size !=
                                                                         'Varies with device'].Size.astype(float).mean()
    enc_df['Size'] = enc_df['Size'].astype(float)

    enc_df['Installs'] = df.Installs.str.strip(
        '+').str.replace(',', '').astype(int)

    enc_df['Type'] = le.fit_transform(df.Type)

    enc_df['Price'] = df.Price.str.strip('$').astype(float)

    enc_df['Content Rating'] = le.fit_transform(df['Content Rating'])

    enc_df['Genres'] = le.fit_transform(df.Genres)
    enc_df['Year'] = enc_df['Last Updated'].map(
            lambda date_string: datetime.datetime.strptime(date_string, "%B %d, %Y").year)

    def DateDiff(date1, date2):
        return (date2 - date1).days

    enc_df['Last Updated'] = df['Last Updated'].apply(pd.to_datetime).map(
        lambda x: DateDiff(x, datetime.datetime.strptime('29062023', "%d%m%Y")))

    enc_df = enc_df.drop(['App', 'Current Ver', 'Android Ver'], axis=1)

    return enc_df


@st.cache_data
def load_review_data():
    review_df = pd.read_csv('data/googleplaystore_user_reviews.csv')
    return review_df


def main():
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Question 1", "Question 2", "Question 3", "Question 4"])
    df = load_data()
    enc_df = preprocess_data(df)
    reviews = load_review_data()

    with tab1:
        st.header("How to increase ratings on the Play Store?")
        col1, col2 = st.columns(2)

        with st.container():
            with col1:
                corr = get_corelations(enc_df)
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                                title="Correlation between attributes")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                feature_importances = get_importances(enc_df)
                fig = px.bar(feature_importances, labels={
                             "index": "Feature", "value": "Importance"})
                fig.update_layout(
                    title="Feature importance of attributes for ratings", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with st.container():
            features = ('Category', 'Reviews', 'Installs', 'Price',
                        'Last Updated', 'Size', 'Content Rating')
            option = st.selectbox('Feature', features)

            if option == 'Category':
                fig = px.box(df, x="Category", y="Rating", color="Category",
                             title="Rating of application in each category is not different too much")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Reviews':
                fig = px.scatter(df.loc[df.Reviews < 10000000], x="Reviews", y="Rating", trendline="ols",
                                 trendline_color_override="red", title="As the number of reviews increase, the rating also increases")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Installs':
                sorted_value = sorted(list(enc_df['Installs'].unique()))
                temp = enc_df.copy()
                temp['Installs'].replace(sorted_value, range(
                    0, len(sorted_value), 1), inplace=True)  # label encide installs
                fig = px.scatter(temp, x="Installs", y="Rating", trendline="ols",
                                 trendline_color_override="red", title="Seem like number of install affect to rating")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Price':
                fig = px.scatter(enc_df, x="Price", y="Rating", trendline="ols",  trendline_color_override="red",
                                 title="Higher price application may make customer disappointed, if they are not good enough")
                st.plotly_chart(fig, use_container_width=True)

                temp = enc_df.copy()
                temp.loc[temp['Price'] == 0, 'PriceBand'] = 'Free'
                temp.loc[(temp['Price'] > 0) & (
                    temp['Price'] <= 0.99), 'PriceBand'] = 'Cheap'
                temp.loc[(temp['Price'] > 0.99) & (
                    temp['Price'] <= 2.99), 'PriceBand'] = 'Not cheap'
                temp.loc[(temp['Price'] > 2.99) & (
                    temp['Price'] <= 4.99), 'PriceBand'] = 'Normal'
                temp.loc[(temp['Price'] > 4.99) & (
                    temp['Price'] <= 14.99), 'PriceBand'] = 'Expensive'
                temp.loc[(temp['Price'] > 14.99) & (temp['Price']
                                                    <= 29.99), 'PriceBand'] = 'Too expensive'
                temp.loc[(temp['Price'] > 29.99),
                         'PriceBand'] = 'Very Very expensive'
                temp[['PriceBand', 'Rating']].groupby(
                    ['PriceBand'], as_index=False).mean()

                fig = px.box(temp, x="PriceBand", y="Rating", color="PriceBand",
                             title="Price are not effect to rating ,but if it is very expensive, it might get low rating")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Last Updated':
                fig = px.scatter(enc_df, x="Last Updated", y="Rating", trendline="ols", labels={
                                 "Last Updated": "No of days since last update"}, trendline_color_override="red")
                fig.update_layout(
                    title="It seems like ratings of the apps are high when its regualry updated")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Size':
                fig = px.scatter(enc_df, x="Size", y="Rating", trendline="ols",
                                 trendline_color_override="red", labels={"Size": "Size (MB)"})
                fig.update_layout(
                    title="Size of the application does not affect the rating")
                st.plotly_chart(fig, use_container_width=True)
            elif option == 'Content Rating':
                fig = px.box(df[df['Content Rating'] != 'Unrated'], x="Content Rating", y="Rating", color="Content Rating",
                             title="Content Rating not effect too much to rating. Mature applications get slightly lower rating than others while 18+ apps have higher average rating.")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Are there any market opportunities for new apps?")
        col1, col2 = st.columns(2)
        with col1:
            cats = break_into_categories(df)
            fig = px.pie(cats, values=cats.values.reshape(-1,).tolist(),
                         names=cats.index, title='Market Breakdown')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df, y="Rating", x="Category", color="Category",
                         title='50% of apps in the Dating category have a rating lesser than the average rating')
            fig.add_hline(y=df.Rating.mean(),
                          line_color='white', line_dash="dash")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            data = get_price_data(df)
            fig = px.bar(data, x='Category', y='App', color='Type')
            fig.update_layout(xaxis_tickangle=-45,
                              title='No of free and paid apps in each category')
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            temp = enc_df.copy()
            temp.Installs = np.log10(temp.Installs)
            fig = px.box(temp, y="Installs", x="Type", color="Type",
                         labels={"Installs": "log10 (Installs)"})
            fig.update_layout(
                title="Paid apps have a relatively lower number of downloads than free apps")
            new_names = {'0': 'Free', '1': 'Paid'}
            fig.for_each_trace(lambda t: t.update(name=new_names[t.name]))
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            new_df = get_sentiment_data(df, reviews)
            trace1 = go.Bar(
                x=list(new_df.Category[::3])[6:-5],
                y=new_df.Sentiment_Normalized[::3][6:-5],
                name='Negative',
                marker=dict(color='rgb(209,49,20)')
            )

            trace2 = go.Bar(
                x=list(new_df.Category[::3])[6:-5],
                y=new_df.Sentiment_Normalized[1::3][6:-5],
                name='Neutral',
                marker=dict(color='rgb(49,130,189)')
            )

            trace3 = go.Bar(
                x=list(new_df.Category[::3])[6:-5],
                y=new_df.Sentiment_Normalized[2::3][6:-5],
                name='Positive',
                marker=dict(color='rgb(49,189,120)')
            )

            data = [trace1, trace2, trace3]
            layout = go.Layout(
                title='Sentiment analysis of reviews',
                barmode='stack',
                xaxis={'tickangle': -45},
                yaxis={'title': 'Fraction of reviews'}
            )

            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            wc = WordCloud(background_color="black",
                           max_words=200, colormap="Set2")
            stop = stopwords.words('english')
            stop = stop + ['app', 'APP', 'ap', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',
                           'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher', 'use', 'user', 'Iam', 'allowed', 'zoom', 'Translated_Review']

            merged_df = merge_df(df, reviews)
            merged_df['Translated_Review'] = merged_df['Translated_Review'].apply(
                lambda x: " ".join(x for x in str(x).split(' ') if x not in stop))
            merged_df.Translated_Review = merged_df.Translated_Review.apply(
                lambda x: x if 'app' not in x.split(' ') else np.nan)
            merged_df.dropna(subset=['Translated_Review'], inplace=True)

            option = st.selectbox('Category', merged_df.Category.unique())

            free = merged_df.loc[(merged_df.Type == 'Free') & (merged_df.Sentiment == 'Negative') & (
                merged_df.Category == option)]['Translated_Review'].apply(lambda x: '' if x == 'nan' else x)
            wc.generate(''.join(str(free)))
            fig = plt.figure(facecolor='black')
            plt.title('Most common words in negative reviews', color='white')
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig, use_container_width=True)

    with tab3:
        st.header("What are the pricing strategies of the apps?")
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            num_free = enc_df['Type'].tolist().count(0)
            num_paid = enc_df['Type'].tolist().count(1)

            labels = ['Free', 'Paid']
            sizes = [num_free, num_paid]
            colors = ['#0a417a', '#72b4eb']

            # Create pie chart using Plotly
            fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, rotation=90,
                            textinfo='label+percent', pull=[0.2, 0], marker=dict(colors=colors))])

            # Customize layout
            fig.update_layout(title='Distribution of Free vs Paid Apps')

            # Display the pie chart using Streamlit
            st.plotly_chart(fig)

        with col2:
            features = ('Reviews', 'Installs', 'Rating')
            option = st.selectbox('Feature', features)

            # if option == 'Installs':
            # Filter out values more than 100M
            # filtered_df = enc_df[enc_df['Installs'] <= 4000000]

            title = 'Relation between ' + option + ' and Price'
            fig = px.scatter(data_frame=enc_df, x='Price', y=option, title=title, marginal_x='histogram', marginal_y='histogram',
                             color_discrete_sequence=['orange'], height=500)
            st.plotly_chart(fig)

        col3, col4 = st.columns(2)
        with col3:
            df['Price'] = df.Price.str.strip('$').astype(float)
            categories = df['Category'].unique().tolist()
            data = [df[['Category', 'Price']][df['Category'] == c]
                    for c in categories]

            # Create scatter plot using Plotly
            fig = go.Figure()
            for i in range(len(categories)):
                fig.add_trace(go.Scatter(
                    x=data[i]['Category'],
                    y=data[i]['Price'],
                    mode='markers',
                    name=categories[i]
                ))

            # Customize layout
            fig.update_layout(
                xaxis=dict(title='Category', tickangle=45),
                yaxis=dict(title='Price ($)'),
                title='Price Distribution by Category',
                width=900
            )

            # Display the scatter plot using Streamlit
            st.plotly_chart(fig)

        with col4:

            df_grouped = enc_df[enc_df['Price']>0].groupby('Year')['Price'].mean().reset_index(name='Avg_Price')
            fig = go.Figure(go.Scatter(
                x=df_grouped['Year'],
                y=df_grouped['Avg_Price'],
                mode='lines',
                name='Avg_Price'
            ))

            fig.update_layout(
                title='Avg_Price of Paid Apps by Year',
                xaxis_title='Year',
                yaxis_title='Avg_Price'
            )

            st.plotly_chart(fig)

        col5, col6 = st.columns(2)

        with col5:
            # grouped_df = df.groupby(['Category']).size().reset_index(name='Count')
            grouped_df = df.groupby(
                ['Category', 'Type']).size().reset_index(name='Count')
            grouped_df = grouped_df[grouped_df['Type'] == 'Paid']
            grouped_df = grouped_df.sort_values('Count', ascending=True)

            fig = go.Figure(go.Bar(
                x=grouped_df['Count'],
                y=grouped_df['Category'],
                orientation='h'
            ))

            fig.update_layout(
                title='Count of Paid Apps by Categories',
                xaxis_title='Count',
                yaxis_title='Category'
            )

            st.plotly_chart(fig)

        with col6:
            grouped_df = df.groupby(
                ['Category', 'Type']).size().reset_index(name='Count')
            selected_categories = ['GAME', 'FAMILY',
                                   'MEDICAL', 'TOOLS', 'PERSONALIZATION']
            grouped_df = grouped_df[grouped_df['Category'].isin(
                selected_categories)].copy()

            fig = px.sunburst(grouped_df, path=['Category', 'Type'], values='Count', color_continuous_scale=['yellow', 'blue'],
                              hover_data=['Category', 'Type', 'Count'])

            fig.update_traces(textinfo='label+value')
            fig.update_layout(
                title='Count of Paid and Free Apps by Category (Sunburst Chart)')
            st.plotly_chart(fig)

    with tab4:
        st.header("How apps are distributed across various categories?")
        features2 = ('Count', 'Installs', 'Rating', 'Reviews')
        option = st.selectbox('Feature2', features2)
        with st.container():
            

            # print(enc_df)
            if option == 'Count':
                grouped_df = df.groupby(['Category']).size().reset_index(name='Count')

                fig = go.Figure(go.Bar(
                    x=grouped_df['Category'],
                    y=grouped_df['Count'],
                ))

                fig.update_layout(
                    title='Count of Apps by Category',
                    xaxis_title='Category',
                    yaxis_title='Count',
                    width=1400
                )

                st.plotly_chart(fig)

            elif option == 'Rating':
                df_grouped = df.groupby('Category')['Rating'].mean().reset_index(name='Avg_rating')
                fig = go.Figure(go.Bar(
                    x=df_grouped['Category'],
                    y=df_grouped['Avg_rating'],
                ))

                fig.update_layout(
                    title='Avg_rating of Apps by Year',
                    xaxis_title='Category',
                    yaxis_title='Avg_rating',
                    width=1400
                )

                st.plotly_chart(fig)
            
            elif option == 'Reviews':
                df_grouped = df.groupby('Category')['Reviews'].mean().reset_index(name='Avg_Reviews')
                fig = go.Figure(go.Bar(
                    x=df_grouped['Category'],
                    y=df_grouped['Avg_Reviews'],
                ))

                fig.update_layout(
                    title='Reviews of Apps by Year',
                    xaxis_title='Category',
                    yaxis_title='Avg_Reviews',
                    width=1400
                )

                st.plotly_chart(fig)
            elif option == 'Installs':
                df['Installs'] = df.Installs.str.strip('+').str.replace(',', '').astype(int)
                df_grouped = df.groupby('Category')['Installs'].mean().reset_index(name='Avg_Installs')
                fig = go.Figure(go.Bar(
                    x=df_grouped['Category'],
                    y=df_grouped['Avg_Installs'],
                ))

                fig.update_layout(
                    title='Reviews of Apps by Year',
                    xaxis_title='Category',
                    yaxis_title='Avg_Installs',
                    width=1400
                )

                st.plotly_chart(fig)




            





if __name__ == '__main__':
    main()
