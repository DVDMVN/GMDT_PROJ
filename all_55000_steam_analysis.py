import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression

from utils import load_all_55000_steam

all_55000_steam = load_all_55000_steam()

def perform_feature_engineering():
    # Categorizing Price
    bins = [-1, 0, 10, 50, float('inf')]  # -1 to include 0 in the first bin
    labels = ['Free to Play', 'Under $10', '$10-50', 'Over $50']
    all_55000_steam['price_category'] = pd.cut(all_55000_steam['price'], bins=bins, labels=labels)

    # Creating a total review counts feature
    all_55000_steam['total_review_count'] = all_55000_steam['positive_reviews'] + all_55000_steam['negative_reviews']
    all_55000_steam['positive_ratio'] = all_55000_steam['positive_reviews'] / all_55000_steam['total_review_count']

    # Categorizing total review counts
    bins = [-1, 7, 15, 30, 60, 125, 250, 500, 1000, 2000, float('inf')]
    all_55000_steam['total_review_bins'] = pd.cut(all_55000_steam['total_review_count'], bins=bins)

    # Making "ownerss" a categorical ordinal feature
    owners_order = [
            "0 .. 20,000",
            "20,000 .. 50,000",
            "50,000 .. 100,000",
            "100,000 .. 200,000",
            "200,000 .. 500,000",
            "500,000 .. 1,000,000",
            "1,000,000 .. 2,000,000",
            "2,000,000 .. 5,000,000",
            "5,000,000 .. 10,000,000",
            "10,000,000 .. 20,000,000",
            "20,000,000 .. 50,000,000",
            "50,000,000 .. 100,000,000",
            "200,000,000 .. 500,000,000",  
        ]
    all_55000_steam['owners'] = pd.Categorical(all_55000_steam['owners'], categories=owners_order, ordered=True)
perform_feature_engineering()

st.header("🌏Exploring All Steam Games from 2022🌏")

st.write(
    """
    In this page, we will explore the massive 55000 steam games dataset, a dataset which represents a complete snapshot of the steam marketplace in 2022. The data was collected using the Steam public API
     as well as steamspy.com's private API.
    """
)

st.markdown("""
1. [Features of Interest](#features-of-interest)
    * A preface on the unique features for this particular dataset, and some caveats to watch out for.
2. [Feature Explorations](#feature-explorations)
    * The exploration journey used to uncover key insights. Jumping to our dashboard is an option, but we _encourage_ a review of our exploration to really grasp understanding
    of our findings. 
    * This section introduces some basic statistics before branching into tabs for particular feature dives, allowing you to choose specific branches of exploration based on your interests.
3. [Key Insights](#key-insights)
    * A comprehensive list of our most important findings, along with some caveats to consider.
""")

# Insert dashboards:
st.subheader("Features of Interest", divider=True)
st.write(
    """
    Before getting into the analysis, let's first explain our key features and how we intend to utilize them. We must also explain some various caveots regarding the 
    integrity of some particular features.

    For this particular dataset, we get the main advantage of "mass". We have the public detailed analytics for the _entire_ steam games dataset, giving our analysis
    a little more integrity than with the top steam games dataset.
        
    - `positive_reviews` and `negative_reviews`:
        - We have access to the number of positive and negative reviews for each game, which can be a valuable metric for analysis. These are metrics that are not normally public,
        and not available through Steam's public API. It is good to have these, however because they are outsourced, we must trust in algorithms for steamspy to judge whether these 
        figures are accurate or at least more accurate than trivial imputation methods. Using these features, we can reasonably impute a review score, and use that to analyze whether
        there are correlations between user ratings and other features.
        - We will also use these counts as a measure of success, in absence of revenue or an accurate copies sold figure.
    - `owners`:
        - Another metric that is not normally public through Steam's public API. This is a range value estimate for the number of owners a game has.
    - `genres`:
        - Game genres that have been assigned to the game. Each game can be given a combination of multiple genres at once. Some combinations are so common that they are often regarded
        as different genres rather than their sum. We may encode these combinations, or explode out the genres, depending on our analysis.
    """
)

st.write(
    """
    ##### Some important caveats:
    1. Our dataset is a snapshot from November of 2022, making it nearly 2 years old. The games marketplace evolves a lot over the years (as we will find), we should
    consider the possibility that some of these analysis are _already_ outdated.
    2. "owners" is an approximation using steamspy's API. This metric is known to be quite inaccurate, especially for smaller titles. For that reason, we will combine 
    this with a trivial imputation by review count.
    3. We do not have a revenue estimate for this dataset, making success a little harder to measure. For this reason, we will be utilizing two imputed metrics. A trivial
    revenue estimate using price, initial_price, and the total review count. And a trivial success measurement, that being whether the game crosses a certain number of reviews
    on steam.
    """
)

st.write("See our documentation page for further details on this dataset's features set:")

st.page_link("./documentation.py", label="Documentation", icon="📔")

st.subheader("Feature Explorations", divider=True)

st.write("#### General items")
st.write(
    """
        Basic statistics with feature listings:
    """
)

numerical_stats_tab, object_stats_tab = st.tabs(["Numerical Statistics", "Categorical Statistics"])

with numerical_stats_tab:
    st.write("### Numerical feature statistics")
    st.write(all_55000_steam.describe())

with object_stats_tab:
    st.write("### Categorical feature statistics")
    st.write(all_55000_steam[all_55000_steam.select_dtypes(include=['object', 'category']).columns].describe())


def plot_correlation_heatmap() -> plt.Figure:
    fig = plt.figure()
    sns.heatmap(
        data=all_55000_steam.select_dtypes(include='number').drop('app_id', axis=1).corr(),
        annot=True,
        cmap='coolwarm',
        vmax=1,
        vmin=-1,
        square=True
    )
    plt.title('Correlation heatmap of numeric features', y=1.01)
    return fig
st.pyplot(plot_correlation_heatmap())

st.write(
    """
    From our correlation heatmap, we can observe that we have a fair number of highly correlated features.
    - Between positive and negative reviews
    - Between CCU (conccurent players online) and positive reviews as well as negative reviews

    The correlation between CCU and reviews of both types tells us a little about the nature of reviews. Reviews of either kind could be 
    an indicator of success! As mentioned before, we utilize this metric often in this dataset, as we do not have a direct revenue estimator,
    and our owners feature is somewhat unreliable in terms of specifics.
    """
)

review_scores_tab, genres_tab = st.tabs(["Review Scores", "Genres"])
with review_scores_tab:
    st.write("### Review Scores Analysis")

    st.write(
        """
        Within this branch, we want to investigate the following questions:
        - What does the total distribution of reviews look like?
        - Do our reviews have any bearing on our success? Do more reviews give us more success?
        - Does having a good positive to negative review ratio correlate with success?
        """
    )

    st.write(
        """
        We begin with plotting our distributions:
        """
    )
    def plot_positive_and_negative_reviews_distribution() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes: list[plt.Axes] = axes.ravel()
        sns.histplot(
            data=all_55000_steam,
            x='positive_reviews',
            bins=30,
            ax=axes[0]
        )

        sns.histplot(
            data=all_55000_steam,
            x='negative_reviews',
            bins=30,
            ax=axes[1],
        )
        sns.histplot(
            data=all_55000_steam,
            x='total_review_count',
            bins=30,
            ax=axes[2],
        )
        plt.tight_layout()
        return fig
    st.pyplot(plot_positive_and_negative_reviews_distribution())

    st.write(
        """
        For all types of review, we can observe qualitatively that we have some extreme outliers for each. We could have infered this from our summary statistics, 
        standard deviation is an order of magnitude higher than mean, but still, examining these things visually gives us a true sense of the distribution.
        - We have several "top dogs" in terms of review counts.
        - We can further on our first question by asking: how does the rest of our data look?
        """
    )

    def plot_frequency_of_games_with_binned_total_review_counts() -> plt.Figure:
        fig = plt.figure(figsize=(15, 6))

        ax = sns.countplot(
            data=all_55000_steam,
            x='total_review_bins',
            # hue='total_review_bins',
        )
        values = all_55000_steam['total_review_bins'].value_counts(sort=False).values / all_55000_steam['total_review_bins'].__len__()
        values = np.round(values * 100, 2)
        values = values.astype(str)
        values = np.array([f'{v}%' for v in values])

        ax.bar_label(container=ax.containers[0], labels=values)

        plt.title('Frequency of games with binned total review counts')
        plt.xlabel('Binned total review counts')
        plt.ylabel('Count')

        return fig
    st.pyplot(plot_frequency_of_games_with_binned_total_review_counts())

    st.write(
        """
        To resolve the issue of outliers, we encoded our total reviews by binning. The exact binning numbers are not "entirely" significant, we chose metrics that made the data both simple and still comprehensive.
        - In particular, we chose a binning strategy that "exponentially grows" our bin. The range of the next bin increases by a factor of 2, until the final bin which is uncapped.

        This plot tells us a lot about the true distribution of our review counts. In particular, we can note a few key statistics:
        - 🌟Over 9% of all games have over 1000 reviews!🌟 Why this is so significant can be explained with a popular trivial revenue check.
            - One method common amongst game developers to estimate revenue is to take the number of reviews and multiply that by a metric represeting average game sales per review (usually between 40-50) before 
            multiplying by the price of the game. Though its accuracy is in debate, performing this check on the lower side, (40), we can estimate around 40000 sales for the game at 1000 reviews!
            - For a game worth 10 dollars, this is already 400000 in revenue! A success for most developers.

        With positive review counts and negative review counts, we can encode a new column "positive_to_negative" which is the ratio between positive and negative review counts.
        """
    )
    def plot_most_liked_publishers() -> plt.Figure:
        min_total_reviews = 1000
        most_positive_publishers = all_55000_steam.groupby('publishers').agg(positive=('positive_reviews', 'sum'),
                                                                            negative=('negative_reviews', 'sum'),
                                                                            quantity=('name', 'count'))
        most_positive_publishers['total_reviews'] = most_positive_publishers['positive'] + most_positive_publishers['negative']
        most_positive_publishers['positive_ratio'] = most_positive_publishers['positive'] / most_positive_publishers['total_reviews']
        most_positive_publishers['positive_to_negative'] = most_positive_publishers['positive'] / most_positive_publishers['negative']
        most_positive_publishers = most_positive_publishers[most_positive_publishers['total_reviews'] >= min_total_reviews]
        most_positive_publishers = most_positive_publishers.sort_values(by='positive_to_negative', ascending=False)

        num_top_publishers = 20
        fig = plt.figure(figsize=(14, 7))
        sns.barplot(
            data=most_positive_publishers.head(num_top_publishers),
            x='positive_to_negative',
            y='publishers',
            hue='publishers',
            orient='h',
            dodge=False
        )
        plt.title(f"Most liked {num_top_publishers} publishers with at least {min_total_reviews} reviews")
        plt.xlabel("Positive to negative reviews ratio")
        plt.ylabel("Publishers")
        return fig
    st.pyplot(plot_most_liked_publishers())

    st.write(
        """
        By grouping then sorting on this new feature, we can get a listing for the "most liked" publishers, that is, the publishers with the highest positive to negative review count.

        We select on a metric such as "at least 1000 reviews total" in order to filter for those publishers who have very few reviews, but have a high ratio.
        - Selecting on "at least 1000" is somewhat arbitrary, but as noted before, trivially it indicates a high degree of success.

        We can repeat this pattern again for developers
        """
    )

    def plot_most_liked_developers() -> plt.Figure:
        min_total_reviews = 1000
        most_positive_developers = all_55000_steam.groupby('developers').agg(positive=('positive_reviews', 'sum'),
                                                                            negative=('negative_reviews', 'sum'),
                                                                            quantity=('name', 'count'))
        most_positive_developers['total_reviews'] = most_positive_developers['positive'] + most_positive_developers['negative']
        most_positive_developers['positive_ratio'] = most_positive_developers['positive'] / most_positive_developers['total_reviews']
        most_positive_developers['positive_to_negative'] = most_positive_developers['positive'] / most_positive_developers['negative']
        most_positive_developers = most_positive_developers[most_positive_developers['total_reviews'] >= min_total_reviews]
        most_positive_developers = most_positive_developers.sort_values(by='positive_to_negative', ascending=False)

        num_top_developers = 20
        fig = plt.figure(figsize=(14, 7))
        sns.barplot(
            data=most_positive_developers.head(num_top_developers),
            x='positive_to_negative',
            y='developers',
            hue='developers',
            orient='h',
            dodge=False
        )
        plt.title(f"Most liked {num_top_developers} developers with at least {min_total_reviews} reviews")
        plt.xlabel("Positive to negative reviews ratio")
        plt.ylabel("Publishers")
        return fig
    st.pyplot(plot_most_liked_developers())

    st.write(
        """
        Some may recognize a few names on here, namely "David Capello", the author of many great tools, most notable being "Aesprite", a powerful and aesthetic pixel art and animation tool.
        """
    )

    def plot_most_liked_owners() -> plt.Figure:
        most_positive_owner_group = all_55000_steam.groupby('owners').agg(positive=('positive_reviews', 'sum'),
                                                                            negative=('negative_reviews', 'sum'),
                                                                            quantity=('name', 'count'))
        most_positive_owner_group['total_reviews'] = most_positive_owner_group['positive'] + most_positive_owner_group['negative']
        most_positive_owner_group['positive_ratio'] = most_positive_owner_group['positive'] / most_positive_owner_group['total_reviews']
        most_positive_owner_group['positive_to_negative'] = most_positive_owner_group['positive'] / most_positive_owner_group['negative']

        

        fig = plt.figure(figsize=(14, 7))
        sns.barplot(
            data=most_positive_owner_group,
            x='positive_to_negative',
            y='owners',
            hue='owners',
            orient='h',
            dodge=False
        )
        plt.title("Game owner estimations vs positive to negative reviews ratio")
        plt.xlabel("Positive to negative reviews ratio")
        plt.ylabel("Owner estimations")
        return fig
    st.pyplot(plot_most_liked_owners())

    st.write(
        """
        Here we are grouping by owner estimations. By taking our owner estimations as ordinal, we can see a trend emerge in "like ratio" vs owner estimation.
        - 🌟As our "like ratio" increases, so too does our playerbase on average🌟. We can see this trend somewhat fall off as we enter bins of the largest magnitudes, however this does seem to hold
        for bins all the way up to 10 million owners.
        """
    )

    def plot_positive_and_negative_review_counts_regression():
        # Linear regression model
        X = all_55000_steam[['positive_reviews']]
        y = all_55000_steam['negative_reviews']
        model = LinearRegression().fit(X, y)
        predicted_negative_review_count = model.predict(X)

        print(model.coef_)

        # Scatter plot with regression line
        fig = px.scatter(
            all_55000_steam,
            x="positive_reviews",
            y="negative_reviews",
            title=f"Positive vs negative review counts (m={model.coef_[0]:.4f}, b={model.intercept_:.4f})",
            labels={"positive_reviews": "Positive Review Count", "negative_reviews": "Negative Review Count"},
        )
        fig.add_traces(px.line(all_55000_steam, x="positive_reviews", y=predicted_negative_review_count).data)

        return fig
    st.plotly_chart(plot_positive_and_negative_review_counts_regression())

    st.write(
        """
        We can quantify the average ratio between positive and negative reviews for games on steam using a regression model.
        - From this, we can see that, on average, for every positive review we can expect to have around 0.15 of a negative review.
        """
    )

with genres_tab:
    st.write("### Genres Analysis")

    st.write(
        """
        Within this branch, we want to investigate the following questions:
        - What are our most popular genres?
        - What genres correlate most with success?
        """
    )

    st.write(
        """
        
        """
    )

    def plot_top_n_genres_distribution(n = 10):
        genre_frequencies = pd.DataFrame(
            {
                'genres': all_55000_steam['genres'].value_counts().index,
                'count': all_55000_steam['genres'].value_counts().values
            }
        )
        # genre_frequencies
        top_genre_frequencies = genre_frequencies.head(n)

        fig = plt.figure(figsize=(15, 6))
        sns.barplot(
            data=top_genre_frequencies,
            x='genres',
            y='count',
            hue='genres',
            dodge=False,
        )
        plt.title(f"Top {n} genres by frequency")
        plt.xlabel("Genres")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        return fig
        # return top_genre_frequences

    st.pyplot(plot_top_n_genres_distribution())

    st.write(
        """
        We have plotted the top genre combinations by frequency. Because there are so many genre combinations (around 5 thousand), we are only plotting the top 10.
        - Somewhat notable is the frequent occurance of the "Action" genre on this list. Combinations of "Action", "Indie" and "Casual" seem to be the most common in genre.
        """
    )

    def plot_exploded_genre_frequencies() -> plt.Figure:
        genres_exploded = all_55000_steam['genres'].str.split(', ').explode()
        genre_counts = genres_exploded.groupby(genres_exploded).size().reset_index(name='genre_count')

        sorted_genre_counts = genre_counts.sort_values(by='genre_count', ascending=False)

        fig = plt.figure(figsize=(15, 6))
        sns.barplot(
            data = sorted_genre_counts,
            x='genres',
            y='genre_count',
            hue='genres',
            dodge=False
        )
        plt.title("Separated genre frequencies")
        plt.xticks(rotation=90)
        plt.xlabel('Genre')
        plt.ylabel('Count')
        return fig
    st.pyplot(plot_exploded_genre_frequencies())

    st.write(
        """
        Instead of plotting by genre combinations (as is defaulted by the dataset), we "explode" the combinations into their individual parts and plot their frequencies.
        
        By separation, we end up with only 28 unique genres total! We plot all of them and their frequencies here.
        - From this, we can note that "Indie", "Action", and "Casual" are truly our most common genres of game (though "Indie" might be less of a genre and more of type).
        - Also interesting to note is that "Free to Play" is a very uncommon genre, nearly dead last next to "Movie".
        """
    )

    def plot_genres_combination_plots() -> list[plt.Figure, plt.Figure, plt.Figure]:
        def split_genres(x):
            if type(x) is str:
                return x.split(', ')
            else:
                return x
        all_55000_steam['genres_split'] = all_55000_steam['genres'].apply(split_genres)
        all_55000_steam_genres_exploded = all_55000_steam.explode('genres_split')
        avg_revenue_by_genre = all_55000_steam_genres_exploded.groupby('genres_split')['total_review_count'].mean().reset_index()
        avg_revenue_by_genre = avg_revenue_by_genre.sort_values(by='total_review_count', ascending=False)

        fig_1 = plt.figure(figsize=(15, 6))
        sns.barplot(
            data=avg_revenue_by_genre,
            x='total_review_count',
            y='genres_split',
            hue='genres_split',
            orient='h',
            dodge=False
        )
        plt.title("Genres by average total review count")
        plt.xlabel('Average total review count')
        plt.ylabel('Genre')

        avg_revenue_by_genre_group = all_55000_steam.groupby('genres')['total_review_count'].mean().reset_index()
        avg_revenue_by_genre_group = avg_revenue_by_genre_group.sort_values(by='total_review_count', ascending=False)

        fig_2 = plt.figure(figsize=(15, 6))
        sns.barplot(
            data=avg_revenue_by_genre_group.head(20),
            x='total_review_count',
            y='genres',
            hue='genres',
            orient='h',
            dodge=False
        )
        plt.title(f"Top {20} genre combinations by average total review count")
        plt.xlabel('Average total review count')
        plt.ylabel('Genre combination')

        genre_group_sizes = all_55000_steam.groupby('genres').size().reset_index(name='genre_count')
        avg_revenue_by_genre_group_with_freq = pd.merge(avg_revenue_by_genre_group, genre_group_sizes, how='left', on='genres')

        fig_3 = plt.figure(figsize=(15, 6))
        sns.barplot(
            data=avg_revenue_by_genre_group_with_freq.head(20),
            x='genre_count',
            y='genres',
            hue='genres',
            orient='h',
            dodge=False
        )
        plt.title(f"Top {20} genre combinations by average total review count: frequencies")
        plt.xlabel('Count')
        plt.ylabel('Genre combination')
        return fig_1, fig_2, fig_3
    fig_1, fig_2, fig_3 = plot_genres_combination_plots()
    st.pyplot(fig_1)

    st.write(
        """
        We can utilize this new "exploded" singular genre feature to take a look at correlations between genre and success. Here we have plotted genre against average total review counts.
        - Free to Play suddenly emerges as our number two in average total review counts. Though this genre is highly uncommon, it seems that there is a large market for it.
        - We can also see something similar in "Massive Multiplayer" and a few other genres.
        """
    )

    st.pyplot(fig_2)

    st.write(
        """
        Doing the same, but now again with our "combination" of genres, we can note again a start difference between frequency and average total review counts. It seems that the less 
        common genres are not necessarily less successful.
        """
    )

    st.pyplot(fig_3)

    st.write(
        """
        To qualify the last statement, we can keep the ordering, but now plot frequencies in place of average total counts.
        """
    )

st.markdown("Back to [Feature Explorations](#feature-explorations)")

st.subheader("Key Insights", divider=True)

with st.container():
    st.markdown(
        """
        <style>
        .stContainer > div {
            width: 150%;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Your container content here
    st.write(
        """
        From our qualitative analysis, we gathered these key insights:

        #### Regarding reviews: 
        - There seems to be "dominators" in this distribution. A small number of top-performing games account for the majority of reviews.
            - This reflects the industry's competitive nature, where a few titles capture most of the attention and player engagement due to factors like higher budgets, marketing, and established fan bases.
        
        - Suprisingly, over 9% of games have accumulated more than 1000 reviews, an unexpectedly high figure.
            - This could be due to social sharing, niche communities, or successful post-launch updates that maintain or grow player engagement.
            - This indicates a longer tail of the market, some creators are dominators, but there is a long tail of successful games that go unrecognized.

        - A higher "like ratio" tends to correspond with larger player bases.
            - Positive reception often signals quality or enjoyment, attracting more players through word of mouth and organic growth.
        
        
        #### Regarding genres:
        - Our most popular genres seem to be Action, Indie, and Casual.
            - These genres often offer broad appeal, allowing for a variety of gameplay styles and experiences that attract diverse audiences.
        
        - Despite the popularity of certain genres, success in terms of revenue or engagement isn't guaranteed.
            - Success might depend more on execution, marketing, and innovation rather than the genre itself, showing that not all popular genres translate into high-performing games.
        """
    )