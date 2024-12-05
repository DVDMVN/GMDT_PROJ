import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import re

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MultiLabelBinarizer

from utils import load_all_55000_steam

all_55000_steam = load_all_55000_steam()
plt.style.use('ggplot')

def perform_feature_engineering(all_55000_steam) -> pd.DataFrame:
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

    # Converting missed datetime format "MON DD, YYYY"
    pattern_mon_day_year_comma = r'^[A-Za-z]{3} \d{1,2}, \d{4}$'  # Matches "Mon DD, YYYY" format
    mask_mon_day_year_comma = all_55000_steam['release_date'].str.match(pattern_mon_day_year_comma, na=False)
    all_55000_steam.loc[mask_mon_day_year_comma, 'release_date'] = pd.to_datetime(all_55000_steam.loc[mask_mon_day_year_comma, 'release_date'], format='%b %d, %Y')

    # Creating a new feature 'release_year' by binning on release_date
    all_55000_steam['release_date'] = pd.to_datetime(all_55000_steam['release_date'], format='%Y-%m-%d')
    all_55000_steam['release_year'] = all_55000_steam['release_date'].dt.year
    all_55000_steam['release_year'] = all_55000_steam['release_year'].astype("Int64")

    # Creating a revenue estimation using initial_price, price, and review counts
    return all_55000_steam
all_55000_steam = perform_feature_engineering(all_55000_steam)

st.header("🌏Exploring All Steam Games from 2022🌏")

st.write(
    """
    In this page, we will explore the massive 55000 Steam games dataset, a dataset which represents a complete snapshot of the Steam marketplace in 2022. The data was collected using the Steam public API
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
4. [Modeling](#modeling)
    * This section attempts to further quality our insights by using machine learning models to predict success.
""")

st.subheader("Features of Interest", divider=True)
st.write(
    """
    Before getting into the analysis, we will first explain our key features and how we intend to utilize them. We must also explain some various caveots regarding the 
    integrity of some particular features.

    For this particular dataset, we get the main advantage of "mass". We have the public detailed analytics for the _entire_ steam games dataset, giving our analysis
    a little more integrity than with the top steam games dataset.
        
    - `positive_reviews` and `negative_reviews`:
        - We have access to the number of positive and negative reviews for each game, typically this information is not public or available through Steam's API. Likely, steamspy's
        API was used to crawl user information.
        - These counts allow us to estimate a 'review_score' metric in two different ways. We can calculate a "percentage positive" metric taking the number of positive_reviews over the total. We could also use
        a ratio metric, the ratio between positive and negative reviews. In the absence of a revenue feature, we will utilize review counts as a reasonable estimate of success.
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
    1. Our dataset is a snapshot from November of 2022, making it nearly 2 years old. The games marketplace is known to evolve quickly (as we will find), we should
    consider the possibility that some of these analysis are _already_ outdated.
    2. "owners" is an approximation using steamspy's API. This metric is known to be quite inaccurate, especially for smaller titles. For this reason, we will prefer to
    utilize review counts as a measure of success, rather than owners.
    3. As mentioned, we do not have a revenue estimate for this dataset, making success a little harder to measure. We will utilize review counts as a trivial estimation for success.
        - A common method for revenue estimation simply scales the review_count by a metric (around 40-50) and the price. We should keep in mind that this metric is not entirely accurate,
        however reasonable it is.
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

@st.cache_data
def plot_correlation_heatmap() -> go.Figure:
    corr_matrix = all_55000_steam.select_dtypes(include='number').drop('app_id', axis=1).corr()
    corr_matrix = np.round(corr_matrix, 2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True, 
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    fig.update_layout(
        title='Correlation heatmap',
        width=700,
        height=700,
        margin=dict(l=120, r=120, t=100, b=100)
    )
    
    return fig

st.plotly_chart(plot_correlation_heatmap())

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

st.write("### Branch Exploration")

review_scores_tab, release_date_tab, genres_tab = st.tabs(["Review Scores", "Release Date", "Genres"])
with review_scores_tab:
    st.write("### Review Scores Analysis")

    st.write(
        """
        Within this branch, we want to investigate the following questions:
        - What does the total distribution of reviews look like? Is attention divided or dominated?
        - What publishers and developers are most liked?
        - Do our reviews have bearing on our success? How much does having a 'better score' help our games?
        """
    )

    st.write(
        """
        We begin with plotting our distributions:
        """
    )
    
    @st.cache_data
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

    @st.cache_data
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

    @st.cache_data
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
        By grouping on publishers then sorting on this new feature, we can get a listing for the "most liked" publishers, that is, the publishers with the highest 
        positive to negative review count.

        We select on a metric such as "at least 1000 reviews total" in order to filter out publishers who have very few reviews, but have a high ratio.
        - Selecting on "at least 1000" is somewhat arbitrary, but as noted before, it can reasonably be considered a high degree of success. In this sense, we are looking
        at the top 20 most liked 'successful' publishers.

        Investigating these names a little further, we find that 🌟all of them are 'indie publishers🌟!
        - Indie publishers seem to dominate in the "likeability lists". While they do not get as much attention in terms of pure review counts, it seems that they are able
        to produce games that are heavily favored.
        - This could be due to them having a more accessible price point (see [our other analysis](https://gmdtproj.streamlit.app/top_1500_steam_analysis#sales-trends)).
        When spending larger amounts of money, users tend to expect a more quality result. User satisfaction might be amplified by a good game, paired with a lower price.
        - This may also be due to the stronger communal sense that 'indie' publishers have. Indie game's lower budgets might also be foster a sense of connection with their
        audiences.
            - The publisher "ConcernedApe" is actually also the single sole developer of the game "Stardew Valley", which is well known to be a very close-knit community, 
            where interactions with the developer are often [[2](https://thinglabs.io/meet-concernedape-the-master-behind-stardew-valley-and-beyond)].
        
        We can repeat this analysis again, but this time for developers:
        """
    )

    @st.cache_data
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
        We see a similar trend in our most liked successful developers. In fact, some of these names on the list are "the same". Indie developers may use their developer
        name as their publisher name as well, some of them keep it the same.
        - Some may recognize a few names on here, namely "David Capello", the author of many great tools, most notable being "Aesprite", a powerful and aesthetic pixel art and animation tool.
        "Igana Studio" is the publisher name he chooses to use.
        """
    )

    @st.cache_data
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
            - Our games with the most owners might be the dominant AAA games in the market (see [our other analysis](https://gmdtproj.streamlit.app/metacritic_analysis) and our release date branch).
        """
        
    )

    @st.cache_data
    def plot_positive_and_negative_review_counts_regression():
        X = all_55000_steam[['positive_reviews']]
        y = all_55000_steam['negative_reviews']
        model = LinearRegression().fit(X, y)
        predicted_negative_review_count = model.predict(X)

        r_2 = model.score(X, y)

        fig = px.scatter(
            all_55000_steam,
            x="positive_reviews",
            y="negative_reviews",
            title=f"Positive vs negative review counts (m={model.coef_[0]:.4f}, b={model.intercept_:.4f})",
            labels={"positive_reviews": "Positive Review Count", "negative_reviews": "Negative Review Count"},
        )
        fig.add_traces(px.line(all_55000_steam, x="positive_reviews", y=predicted_negative_review_count).data)

        return fig, r_2
    fig, r_2 = plot_positive_and_negative_review_counts_regression()
    st.plotly_chart(fig)

    st.write(
        """
        We can quantify the average ratio between positive and negative reviews for games on steam using a regression model.
        - From this, we can see that, on average, for every positive review we can expect to have around 0.15 of a negative review.
        - This is one of our stronger correlations (R = 0.78), we can achieve an $R^2$ value of {r_2:.2f}
        """.format(r_2=r_2)
    )


with release_date_tab:
    st.write("### Review Scores Analysis")
    st.write(
        """
        Within this branch, we want to investigate the following questions:
        - Is the game market becoming more one sided? Is it becoming more winner take all over the years?
        - How has the popularity of genres varied with time?
        """
    )

    st.write(
        """
        The get a lay of the snapshot timespan, we plot the distribution of game releases per year.
        - To smooth the release date distribution a little, we binned the release dates into years.
        """
    )

    @st.cache_data
    def plot_release_date_distribution() -> plt.Figure:
        fig = plt.figure(figsize=(12, 6))
        sns.countplot(
            data=all_55000_steam,
            y='release_year',
            orient='h',
        )
        plt.title("Game release year frequency")
        plt.xlabel("Count")
        plt.ylabel("Release year")
        return fig
    st.pyplot(plot_release_date_distribution())

    st.write(
        """
        From this, we can see that most of our games are from recent years. Though Steam has existed for a long time (Since Sept. 12 2003), they often retire very old games as new games come in.
        - Other game market platforms, such as GOG, do not have such practice. Buying a game on Steam has a clause that it may be eventually retired.

        From here, we can investigate a question; whether games from past years see more or less total reviews than games released nearer to the present.
        - Increases or decreases in total reviews might tell us something about the market interest in newer games.
        """
    )

    @st.cache_data
    def plot_release_year_to_total_reviews() -> plt.Figure:
        fig = plt.figure(figsize=(12, 6))
        reviews_by_year = all_55000_steam.groupby("release_year")['total_review_count'].sum()
        sns.barplot(
            x=reviews_by_year.values,
            y=reviews_by_year.index,
            orient='h',
        )
        plt.xlabel('Total review count')
        plt.ylabel('Release year')
        plt.title('Total review counts on Steam by year')
        return fig
    st.pyplot(plot_release_year_to_total_reviews())

    st.write(
        """
        Evidently, we do not have a completely straightforward answer. From this plot we can see a few things:
        - The total number of reviews from all games released in 2012 outnumbers all other years except 2017.
            - Perhaps there were some major releases from this era that became timeless collection pieces. There may be an element of nostalgia for
            older gamers present in this market.
            - 2012 saw the release of several breakout consoles, such as the PS4 and XBox One. This spike in reviews could be indicative of a
            gaming renaissance, stronger engagment in the game market due to major leaps forward in innovation and quality.
        """
    )
    
    def plot_top_10_sum_and_proportion() -> plt.Figure:
        sorted_review_count_per_year = all_55000_steam.sort_values(by='total_review_count', ascending=False)
        total_review_count_per_year = sorted_review_count_per_year.groupby("release_year")[['name', 'total_review_count']].agg({
            'name': 'count',
            'total_review_count': 'sum',
        }).rename(
            columns={'name': 'count'}
        ).reset_index()

        total_review_count_per_year = total_review_count_per_year[total_review_count_per_year['count'] > 1000]

        total_review_count_per_year_top_10 = (
            sorted_review_count_per_year.groupby("release_year")
            .head(10)
            .groupby("release_year")
            .agg({
                'total_review_count': 'sum'
            }).rename(
                columns={'total_review_count': 'total_review_count_top_10'}
            )
            .reset_index()
        )

        merged_total_and_slicesum_review_count_per_year = pd.merge(total_review_count_per_year, total_review_count_per_year_top_10, on='release_year', how='left')
        merged_total_and_slicesum_review_count_per_year[
            "proportion_of_reviews_of_top_10_to_total"
        ] = (
            merged_total_and_slicesum_review_count_per_year["total_review_count_top_10"]
            / merged_total_and_slicesum_review_count_per_year["total_review_count"]
        )
        # print(merged_total_and_slicesum_review_count_per_year)
        fig = plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=merged_total_and_slicesum_review_count_per_year,
            x='release_year',
            y='total_review_count_top_10',
            dodge=False,
        )
        values = merged_total_and_slicesum_review_count_per_year['proportion_of_reviews_of_top_10_to_total']
        values = np.array([f'{v * 100:.2f}%' for v in values])
        ax.bar_label(container=ax.containers[0], labels=values)
        ax.text(x=0.5, y=1.05, s="With percentage to the year's total review count", fontsize=10, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
        plt.title("Total review counts for the top 10 games by year", y=1.1)
        plt.xlabel("Release year")
        plt.ylabel("Total reviews of top 10")
        return fig
    
    st.pyplot(plot_top_10_sum_and_proportion())

    st.write(
        """
        From this distribution graph, we are plotting the total review counts for the top 10 most reviewed games of each year. 
        We also indicate the proportion of the top 10's sum of reviews to that year's total review count.

        This tells us a lot about market concentration over the years. For games in 2015, for example, with the top 10 taking already more than half of the review counts, this could be indicative that
        2015 in particular had some breakout games which hogged the limelight.
        - 2015 saw the release of some of the most highly acclaimed games of all time. A few that come to mind are "The Witcher 3", "Metal Gear Solid V" and "Bloodborne"; all of which have already been
        immortalized within game culture.
            - 🌟Its possible that some years see domination by large budget AAA publishers, leaving little room for smaller, indie studios to capture attention!🌟

        From these values, we can observe that the relation between release year and top 10 proportion to the market is not clear (at least not clearly linear). From one year to the next we can see 
        jumps up and down. Some years see much larger domination than others, and it may be difficult to forecast coming years from this information alone.
        """
    )

    @st.cache_data
    def plot_bubble_proportions_of_genre_by_year() -> plt.Figure:
        all_55000_steam_copy = all_55000_steam.copy(deep=True)
        all_55000_steam_copy['genres'] = all_55000_steam_copy['genres'].str.split(', ')
        all_55000_steam_exploded_genres = all_55000_steam_copy.explode('genres')
        genre_counts = all_55000_steam_exploded_genres.groupby(['release_year', 'genres']).size().reset_index(name='count')
        total_genres_per_year = all_55000_steam_exploded_genres.groupby('release_year').size().reset_index(name='total_genres')
        genre_proportions = pd.merge(genre_counts, total_genres_per_year, on='release_year')

        genre_proportions_filtered = genre_proportions[(genre_proportions['total_genres'] > 1000) & 
                                                    (genre_proportions['count'] >= 60)]
        genre_proportions_filtered['genre_proportion'] = genre_proportions_filtered['count'] / genre_proportions_filtered['total_genres']


        fig = plt.figure(figsize=(12, 8))
        plt.scatter(
            x = genre_proportions_filtered["release_year"],
            y = genre_proportions_filtered["genres"],
            s=genre_proportions_filtered["genre_proportion"] * 5000,  # Scaling the bubble size by proportion * number for clarity
            alpha=0.6,
            edgecolors="w",
            linewidth=2,
        )

        plt.xlabel('Release year')
        plt.ylabel('Genre')
        plt.title('Proportion of genres by year')
        plt.xticks(rotation=45)
        plt.grid(True)
        return fig
    st.pyplot(plot_bubble_proportions_of_genre_by_year())

    st.write(
        """
        From here, we can try to investigate whether genre has been varying through the years. Here we have plotted some of the more popular genres against the release year, with the bubble's size indicating
         the proportion of that genre to all other games in that year.

        We can see that "Indie" and "Action" games have been significant proportions of the game marketplace consistently for the past decade. In fact, most genres seem to be quite stable in their proportion to the market.
        - Notably, however, we can see that "Free to Play" games have been falling down as of recent years. We may want to investigate this further.
        - Looking at how genres rising and decline over the years may give us a sense of how player preferences shift. If we can identify key factors which may allow us to predict future genre success, we may tailor
        our games to specific types of gameplay.
        """
    )

    @st.cache_data
    def plot_genres_of_interest_trends() -> plt.Figure:
        all_55000_steam_copy = all_55000_steam.copy(deep=True)
        all_55000_steam_copy['genres'] = all_55000_steam_copy['genres'].str.split(', ')
        all_55000_steam_exploded_genres = all_55000_steam_copy.explode('genres')
        genre_counts = all_55000_steam_exploded_genres.groupby(['release_year', 'genres']).size().reset_index(name='count')
        total_genres_per_year = all_55000_steam_exploded_genres.groupby('release_year').size().reset_index(name='total_genres')
        genre_proportions = pd.merge(genre_counts, total_genres_per_year, on='release_year')

        genre_proportions_filtered = genre_proportions[(genre_proportions['total_genres'] > 1000) & 
                                                    (genre_proportions['count'] >= 60)]

        genres_of_interest = ["Free to Play", "Indie", "Strategy"]
        genre_trends = genre_proportions_filtered[genre_proportions_filtered['genres'].isin(genres_of_interest)]

        fig = plt.figure(figsize=(10, 6))

        for genre in genres_of_interest:
            genre_data = genre_trends[genre_trends['genres'] == genre]
            plt.plot(genre_data['release_year'], genre_data['count'], label=genre, marker='o')

        plt.xlabel('Release year')
        plt.ylabel('Count')
        plt.title('Popularity of Free to Play, Indie, and Strategy games over years')
        plt.xticks(rotation=45)
        plt.legend(title="Genres")
        plt.grid(True)
        return fig
    st.pyplot(plot_genres_of_interest_trends())

    st.write(
        """
        Here we have plotted the popularity of a select few genres over the last decade. From this, we can see that although "Indie" has been relatively similar in proportion through the years, it has grown in size considerably
        as the general game market continues to grow. We might expect indie games to continue to be prevalent in the market.

        We can observe the dip in "Free to Play" releases from the last 2 years 2020 and 2021. This is interesting, as from other analysis prior, we learned the "Free to Play" games actually take a very sizeable portion of the 
        marketplace revenue.
        - This genre is becoming rarer, but still has a very sizeable stake in the marketplace in terms of success!

        By examining these trends, we might get a better sense of the current player demographics, where they originate from, and what current opportunities we are missing.
        """
    )

with genres_tab:
    st.write("### Genres Analysis")

    st.write(
        """
        Within this branch, we want to investigate the following questions:
        - What the genre distribution look like?
        - What are our most popular genres?
        - Can we find any genres that, in particular, games at the top seem to take?
        """
    )

    st.write(
        """
        Because our genre feature is actually the game's "combination of genres", we will start with plotting our genre combinations.
        - In some sense, a mixture of genres can be considered a different genre on its own. For this reason, it is still insightful to note genre 
        combinations as well as the individualized genres.
        """
    )

    @st.cache_data
    def plot_top_n_genres_distribution(n = 10):
        genre_frequencies = pd.DataFrame(
            {
                'genres': all_55000_steam['genres'].value_counts().index,
                'count': all_55000_steam['genres'].value_counts().values
            }
        )
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

    st.pyplot(plot_top_n_genres_distribution())

    st.write(
        """
        We have plotted the top genre combinations by frequency. Because there are so many genre combinations (around 5 thousand), we are only plotting the top 10.
        - Somewhat notable is the frequent occurance of the "Action" genre on this list. Combinations of "Action", "Indie" and "Casual" seem to be the most common in genre.

        By "exploding" our genre combinations (separating to individual), we can look closer at how the individual genres are distributed:
        """
    )

    @st.cache_data
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
        After "explosion", we end up with only 28 unique genres total.
        - From our distribution, we can note that "Indie", "Action", and "Casual" are indeed our most common genres of game.
        - Evidently, Steam allows for many "non-game' related genres. We have genres like "Photo Editing", "Design & Illustration", "Movie", etc. Steam does provide a platform for various
        software that are not games, however, just from this distribution on genres, we can know that these applications are very uncommon.
            - We actually have very few 'game descriptive' genres, that is, genres that describe the gameplay of our games. Game developers looking to target a niche community should not
            look to do so from this listing.
        """
    )

    @st.cache_data
    def plot_genres_combination_plots() -> plt.Figure:
        def split_genres(x):
            if type(x) is str:
                return x.split(', ')
            else:
                return x
        all_55000_steam['genres_split'] = all_55000_steam['genres'].apply(split_genres)
        all_55000_steam_genres_exploded = all_55000_steam.explode('genres_split')
        avg_revenue_by_genre = all_55000_steam_genres_exploded.groupby('genres_split')['total_review_count'].mean().reset_index()
        avg_revenue_by_genre = avg_revenue_by_genre.sort_values(by='total_review_count', ascending=False)

        fig = plt.figure(figsize=(15, 6))
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
        return fig
    st.pyplot(plot_genres_combination_plots())

    st.write(
        """
        Here we have plotted the average total review count for each individual genre.
        - We can see that some of the rarer genres, such as "Massively Multiplayer", or "Free to Play", have suprisingly great average review counts despite their frequency.
            - These rarer frequency games are somewhat 'outliers' and therefore may not give us a real assessment of the genre as a whole. Without sufficient frequency within a genre, our
            analysis lacks integrity.
            - 🌟This distribution however does give us a sense of potential success🌟. It shows that rarer genre mixtures can potentially reach higher charts than the averages of other genres.
        """
    )

    @st.cache_data
    def plot_genre_frequencies_with_success_rate() -> plt.Figure:
        n = 500
        genre_group_sizes = all_55000_steam.groupby('genres').size().reset_index(name='genre_count')
        genre_total_reviews = all_55000_steam.groupby('genres')['total_review_count'].median().reset_index(name='average_total_review_count')
        genres_with_at_least_n_games = genre_group_sizes[genre_group_sizes['genre_count'] >= 500]['genres'].tolist()
        # print(genres_with_at_least_n_games.__len__())

        all_55000_steam_within_genre_list = all_55000_steam[all_55000_steam['genres'].isin(genres_with_at_least_n_games)]
        all_55000_steam_within_genre_list = pd.merge(all_55000_steam_within_genre_list, genre_total_reviews, on='genres', how='left')
        all_55000_steam_within_genre_list['over_1000_total_reviews'] = all_55000_steam_within_genre_list['total_review_count'] > 1000
        all_55000_steam_within_genre_list = all_55000_steam_within_genre_list.sort_values(by='genres')

        stacked_data = all_55000_steam_within_genre_list.groupby(['genres', 'over_1000_total_reviews']).size().unstack()
        stacked_data['total_count'] = stacked_data.sum(axis=1)
        stacked_data = stacked_data.reset_index()
        stacked_data.index.name = None
        stacked_data['percentage_successful'] = (stacked_data[True] / stacked_data['total_count']) * 100
        stacked_data['percentage_successful'] = stacked_data['percentage_successful'].round(2)

        fig = plt.figure(figsize=(12, 8))

        ax = sns.histplot(
            data=all_55000_steam_within_genre_list,
            y='genres',
            hue='over_1000_total_reviews',
            multiple='stack',
        )

        ax.bar_label(ax.containers[0], labels=[" " + str(x) + '% True' for x in stacked_data['percentage_successful']])
        ax.bar_label(ax.containers[1], labels=["  " + str(x) for x in stacked_data['total_count']])
        ax.legend_.set_title("Success status")
        for text, label in zip(ax.legend_.texts, ['No', 'Yes']):
            text.set_text(label)
        plt.title("Popular genres frequencies and their success rates", y=1.07)
        ax.text(x=0.5, y=1.03, s=f"Popular = with at least {n} games | Success = with over 1000 total reviews", fontsize=10, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
        plt.xlabel("Count")
        plt.ylabel("Genre(s)")

        return fig
    st.pyplot(plot_genre_frequencies_with_success_rate())

    st.write(
        """
        Here we plot only genre combinations with some higher degree of integrity. We also differentiate by genre combinations instead of individual genres; with our limited set of gameplay descriptors, 
        combinations give us a better specification than the parts. 
        - We define success by having over 1000 total reviews (see notes from before), and we also are filtering for popular genres, in this case meaning
        at least 500 games were made with the same genre.
        - Having at least 500 is an arbitraryish benchmark for "common enough". With a dataset of 50000 +, 500 felt like a reasonable "1%", which could give our analysis on these genres some degree of integrity.

        From this we can qualitatively observe certain genre combinations having better success rates than others. We can also see that certain genre combinations have commonalities in parts.
        - Certain genres, like RPGs and strategy games, have higher success rates. Perhaps they appeal more to players, or at least the established expectations are easier to reach.
        - The low success of games with only the indie genre and the relatively high rate of games that are indie suggests that indie games require some 'other factors' to stand out. Perhaps indie games require some emphasis
        on stronger or unique gameplay features.
        """
    )

    # def plot_genres_combination_plots() -> list[plt.Figure, plt.Figure, plt.Figure]:
    #     def split_genres(x):
    #         if type(x) is str:
    #             return x.split(', ')
    #         else:
    #             return x
    #     all_55000_steam['genres_split'] = all_55000_steam['genres'].apply(split_genres)
    #     all_55000_steam_genres_exploded = all_55000_steam.explode('genres_split')
    #     avg_revenue_by_genre = all_55000_steam_genres_exploded.groupby('genres_split')['total_review_count'].mean().reset_index()
    #     avg_revenue_by_genre = avg_revenue_by_genre.sort_values(by='total_review_count', ascending=False)

    #     fig_1 = plt.figure(figsize=(15, 6))
    #     sns.barplot(
    #         data=avg_revenue_by_genre,
    #         x='total_review_count',
    #         y='genres_split',
    #         hue='genres_split',
    #         orient='h',
    #         dodge=False
    #     )
    #     plt.title("Genres by average total review count")
    #     plt.xlabel('Average total review count')
    #     plt.ylabel('Genre')

    #     avg_revenue_by_genre_group = all_55000_steam.groupby('genres')['total_review_count'].mean().reset_index()
    #     avg_revenue_by_genre_group = avg_revenue_by_genre_group.sort_values(by='total_review_count', ascending=False)

    #     fig_2 = plt.figure(figsize=(15, 6))
    #     sns.barplot(
    #         data=avg_revenue_by_genre_group.head(20),
    #         x='total_review_count',
    #         y='genres',
    #         hue='genres',
    #         orient='h',
    #         dodge=False
    #     )
    #     plt.title(f"Top {20} genre combinations by average total review count")
    #     plt.xlabel('Average total review count')
    #     plt.ylabel('Genre combination')

    #     genre_group_sizes = all_55000_steam.groupby('genres').size().reset_index(name='genre_count')
    #     avg_revenue_by_genre_group_with_freq = pd.merge(avg_revenue_by_genre_group, genre_group_sizes, how='left', on='genres')

    #     fig_3 = plt.figure(figsize=(15, 6))
    #     sns.barplot(
    #         data=avg_revenue_by_genre_group_with_freq.head(20),
    #         x='genre_count',
    #         y='genres',
    #         hue='genres',
    #         orient='h',
    #         dodge=False
    #     )
    #     plt.title(f"Top {20} genre combinations by average total review count: frequencies")
    #     plt.xlabel('Count')
    #     plt.ylabel('Genre combination')
    #     return fig_1, fig_2, fig_3
    # fig_1, fig_2, fig_3 = plot_genres_combination_plots()
    # st.pyplot(fig_1)

    # st.write(
    #     """
    #     We can utilize this new "exploded" singular genre feature to take a look whether there is a trend between between genre and success. Here we have plotted genre against average total review counts.
    #     - Free to Play suddenly emerges as our number two in average total review counts. Though this genre is highly uncommon, it seems that there is a large market for it.
    #     - We can also see something similar in "Massive Multiplayer" and a few other genres.
    #     """
    # )

    # st.pyplot(fig_2)

    # st.write(
    #     """
    #     Doing the same, but now again with our "combination" of genres, we can note again a start difference between frequency and average total review counts. It seems that the less 
    #     common genres are not necessarily less successful.
    #     """
    # )

    # st.pyplot(fig_3)

    # st.write(
    #     """
    #     To qualify the last statement, we can keep the ordering, but now plot frequencies in place of average total counts.
    #     """
    # )

st.markdown("[Explore other branches](#branch-exploration)")

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
    with st.container(border=True):
        st.write(
            """
            **🌟Achievable Success🌟**
            """
        )
        def plot_frequency_of_games_with_binned_total_review_counts() -> plt.Figure:
            fig = plt.figure(figsize=(15, 6))

            ax = sns.countplot(
                data=all_55000_steam,
                x='total_review_bins',
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
            - Over 9% of games can reach and surpass 1000 total reviews!
                - This matches a finding by VGInsights on their analysis of the steam market in 2020 [[4](https://vginsights.com/insights/article/infographic-indie-game-revenues-on-steam)].
            - Reaching success is feasible for many creators; dominators do not pull all the success away from smaller creators.
            """
        )

    # Genres
    with st.container(border=True):
        st.write(
            """
            **🌟Adapting Genre Trends🌟**
            """
        )

        pick_from_genre_list = ['Massively Multiplayer', 'Early Access', 'Racing', 'Free to Play', 'Strategy', 'RPG', 'Indie', 'Casual', 'Adventure', 'Action']

        @st.fragment()
        def plot_genres_of_interest_trends_plotly():
            genres_of_interest = st.multiselect("Pick genres to plot:", pick_from_genre_list, default=['Free to Play', 'Indie', 'Strategy'])
            all_55000_steam_copy = all_55000_steam.copy(deep=True)
            all_55000_steam_copy['genres'] = all_55000_steam_copy['genres'].str.split(', ')
            all_55000_steam_exploded_genres = all_55000_steam_copy.explode('genres')
            
            
            genre_counts = all_55000_steam_exploded_genres.groupby(['release_year', 'genres']).size().reset_index(name='count')
            total_genres_per_year = all_55000_steam_exploded_genres.groupby('release_year').size().reset_index(name='total_genres')
            
            
            genre_proportions = pd.merge(genre_counts, total_genres_per_year, on='release_year')
            
            
            genre_proportions_filtered = genre_proportions[(genre_proportions['total_genres'] > 1000) & 
                                                        (genre_proportions['count'] >= 60)]
            
            
            genre_trends = genre_proportions_filtered[genre_proportions_filtered['genres'].isin(genres_of_interest)]
            colors = pc.qualitative.Plotly
            
            fig = px.line(
                genre_trends,
                x='release_year',
                y='count',
                color='genres',
                title='Popularity of select game genres over time',
                labels={'count': 'Count', 'release_year': 'Release year'},
                markers=True,
                color_discrete_sequence=colors
            )

            fig.update_layout(
                xaxis_title='Release year',
                yaxis_title='Count',
                legend_title_text='Genres',
                width=900,
                height=600,
            )
            
            st.plotly_chart(fig)
        plot_genres_of_interest_trends_plotly()

        st.write(
            """
            - Indie games, though they have always been quite dominant in the marketplace (in terms of frequency), they are growing in size, staying in proportion to the growing game market. As the 
            market has grown, more than ever small creators have been creating and releasing games.
            - Certain other trends, such as "Free to Play" being in the decline, could represent opportunities for developers to capture certain untapped markets, so long as there is still a market to 
            be had.
            """
        )

    with st.container(border=True):
        st.write(
            """
            **🌟Genre Success Rates🌟**
            """
        )

        @st.fragment()
        def dashboard_plot_genre_frequencies_and_success_status():
            at_least_n_games = st.slider(
                "Define popular genre game threshold (at least 'this' many games in genre to be popular):", min_value=300, max_value=1500, value=500, step=25
            )
            n = at_least_n_games
            genre_group_sizes = all_55000_steam.groupby('genres').size().reset_index(name='genre_count')
            genre_total_reviews = all_55000_steam.groupby('genres')['total_review_count'].median().reset_index(name='average_total_review_count')
            genres_with_at_least_n_games = genre_group_sizes[genre_group_sizes['genre_count'] >= n]['genres'].tolist()
            # print(genres_with_at_least_n_games.__len__())

            all_55000_steam_within_genre_list = all_55000_steam[all_55000_steam['genres'].isin(genres_with_at_least_n_games)]
            all_55000_steam_within_genre_list = pd.merge(all_55000_steam_within_genre_list, genre_total_reviews, on='genres', how='left')
            all_55000_steam_within_genre_list['over_1000_total_reviews'] = all_55000_steam_within_genre_list['total_review_count'] > 1000
            all_55000_steam_within_genre_list = all_55000_steam_within_genre_list.sort_values(by='genres')

            stacked_data = all_55000_steam_within_genre_list.groupby(['genres', 'over_1000_total_reviews']).size().unstack()
            stacked_data['total_count'] = stacked_data.sum(axis=1)
            stacked_data = stacked_data.reset_index()
            stacked_data.index.name = None
            stacked_data['percentage_successful'] = (stacked_data[True] / stacked_data['total_count']) * 100
            stacked_data['percentage_successful'] = stacked_data['percentage_successful'].round(2)

            colors = pc.qualitative.Plotly

            fig = px.histogram(
                    all_55000_steam_within_genre_list,
                    y='genres',
                    color='over_1000_total_reviews',
                    barmode='stack',
                    histfunc='count',
                    labels={'over_1000_total_reviews': 'Success status', 'genres': 'Genres'},
                    color_discrete_map={True: colors[0], False: colors[1]} 
                )

            success_status_name_changes = {
                'True': 'Yes',
                'False': 'No',
            }
            fig.for_each_trace(lambda t: t.update(name = success_status_name_changes[str(t.name)],
                                                legendgroup = success_status_name_changes[str(t.name)],
                                                hovertemplate = t.hovertemplate.replace(str(t.name), success_status_name_changes[str(t.name)])
                                                )
            )

            # Hacky way to add annotations to the bars, since I have a 'stack' mode histogram, I can just add a 'transparent bar' to each in same order with text.
            for i, row in stacked_data.iterrows():
                fig.add_trace(go.Bar(
                    y=[row['genres']],
                    x=[10] * len(stacked_data), # Text spacing of 10 count
                    orientation='h',
                    showlegend=False,
                    text=f"{row['percentage_successful']}%",
                    textfont_size=15,
                    marker_color='rgba(0,0,0,0)',  # Hide bar
                    hoverinfo='skip'
                ))

            fig.update_layout(
                yaxis_title="",
                xaxis_title="Count",
                legend_title="Success status",
                bargap=0.1,
                height=600,
                showlegend=True,
                margin=dict(l=0, r=0, t=50, b=50),
                title=go.layout.Title(
                    text=f"Popular genres frequencies and their success rates<br><sup>Popular = with at least {n} games | Success = with over 1000 total reviews</sup>"
                ),
                legend=dict(
                    x=0.85,
                    y=0.95
                )
            )
            st.plotly_chart(fig)
        dashboard_plot_genre_frequencies_and_success_status()

        st.write(
            """
            - Popularity of a genre does not determine success rates.
            - Popular indie game combinations games tend to have better success rates than games with only the individual indie genre.
            - We can see RPGs and strategy games are popular and have higher success rates. Perhaps the concepts of RPG and strategy appeal more to players [[3](https://nitemare121.medium.com/why-are-rpgs-popular-a-stunning-world-awaits-3f629adfa1b)]
            , or at least the established expectations are easier to reach. There are likely many confounding factors at play which make certain genre more successful on average than others.
            """
        )

st.subheader("Modeling", divider=True)

st.write(
    """
    Within the modeling section, we showcase our attempt at training, testing and validating of machine learning models to predict game success. 
    We are predict values of 'total_review_count' as our metric of success. We will refrain from using trivial correlators like 'positive_reviews' or 'owners' to 
     ensure integrity of our analysis. Instead we will be using the following features:
    - `developer_experience`
    - `publisher_experience`
    - `genre`
    - `languages`
    - `languages_supported`

    With our primary interest being in `developer_experience`, with the rest as features to control their potential confounding effect on the 
    outcome and predictor.
    - We want to attempt to shed light on a popular question: "does experience really matter for game development?". Sometimes refered to as "DX" in the industry space, we
    often see job recruiters place heavy emphasis on prior experience for potential hiring candidates. We would like to quantify its predictive effect.

    To do this, we will be utilizing the following three models:
    - `LinearRegression`
    - `RandomForest`
    - `XGBoost`

    These models were chosen to compare performance at differing orders of complexity and to compare differing methods to capture a best fit.
    - The multivariate linear regression model is associated heavily with linear relationships. Though we found very few great linear correlators in our
    correlation heatmap assessment, we will utilize this model to baseline the others. 
        - Note: Having little linear correlation with the target variable brings into light interpretability issues. Our derived weights for our features lose integrity in telling of
        the features' true effect.
    - Though random forest models are often associated with classification tasks, they can generalize to regression tasks as well. Random forest and decision tree models 
    do a much better job at capturing non-linearity in the data. 
    - XGBoost, another derivative of decision tree model, utilizes a differing algorithm in tree construction. Due to its sequential "boosting", feature importances are
    lose integrity, though they may still be extracted [[16](https://mljourney.com/xgboost-feature-importance-comprehensive-guide/)].
    """
)

status_message = "Running Models..."

st.write(status_message)
# The additional feature_engineering is specifically for the modeling portion of the analysis
def perform_additional_feature_engineering(all_55000_steam) -> pd.DataFrame:
    # Getting total developer and publisher experiences
    filter_terms = ["Inc", "Inc.", "LLC", "Ltd", "Ltd.", "LTD."] # The list of developers includes lots of filler stuff that throws off our count. Not all "LLC" work on the same games after all.
    
    all_55000_steam['developers'] = all_55000_steam['developers'].fillna('').astype(str)
    all_55000_steam['publishers'] = all_55000_steam['publishers'].fillna('').astype(str)

    def filter_entities(entity_list):
        return [entity for entity in entity_list if not any(term in entity for term in filter_terms)]

    all_55000_steam['developer_list'] = all_55000_steam['developers'].apply(lambda x: [dev.strip() for dev in x.split(',') if dev.strip()])
    all_55000_steam['publisher_list'] = all_55000_steam['publishers'].apply(lambda x: [dev.strip() for dev in x.split(',') if dev.strip()])

    all_55000_steam['developer_list'] = all_55000_steam['developer_list'].apply(filter_entities)
    all_55000_steam['publisher_list'] = all_55000_steam['publisher_list'].apply(filter_entities)

    developer_df = all_55000_steam[['app_id', 'developer_list']].explode('developer_list')
    publisher_df = all_55000_steam[['app_id', 'publisher_list']].explode('publisher_list')

    developer_counts = developer_df.groupby('developer_list')['app_id'].nunique().reset_index()
    developer_counts.rename(columns={'app_id': 'developer_experience'}, inplace=True)
    publisher_counts = publisher_df.groupby('publisher_list')['app_id'].nunique().reset_index()
    publisher_counts.rename(columns={'app_id': 'publisher_experience'}, inplace=True)

    developer_df = developer_df.merge(developer_counts, on='developer_list', how='left')
    publisher_df = publisher_df.merge(publisher_counts, on='publisher_list', how='left')

    dev_exp_avg = developer_df.groupby('app_id')['developer_experience'].sum().reset_index()
    all_55000_steam = all_55000_steam.merge(dev_exp_avg, on='app_id', how='left')

    pub_exp_avg = publisher_df.groupby('app_id')['publisher_experience'].sum().reset_index()
    all_55000_steam = all_55000_steam.merge(pub_exp_avg, on='app_id', how='left')

    # Exploding genres using a MultiLabelBinarizer
    all_55000_steam['genres'].fillna("", inplace=True)
    all_55000_steam['genres_list'] = all_55000_steam['genres'].str.split(', ')

    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(all_55000_steam['genres_list'])

    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    all_55000_steam = pd.concat([all_55000_steam, genres_df], axis=1)
    all_55000_steam = all_55000_steam.drop('genres_list', axis=1)

    # Exploding price category using pd.get_dummies
    price_category_dummies = pd.get_dummies(all_55000_steam['price_category'], prefix='price_category')
    all_55000_steam = pd.concat([all_55000_steam, price_category_dummies], axis=1)

    all_55000_steam['languages'].fillna("", inplace=True) # if the game doesnt have it listed, then it is simply left blank.
    possible_languages = ['Afrikaans', 'Albanian', 'Arabic', 'Azerbaijani', 'Bangla', 'Basque', 'Belarusian',
                        'Bosnian', 'Bulgarian', 'Catalan', 'Croatian', 'Czech', 'Danish', 'Dari', 'Dutch',
                        'English', 'Estonian', 'Filipino', 'Finnish', 'French', 'Galician', 'Georgian',
                        'German', 'Greek', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian',
                        'Irish', 'Italian', 'Japanese', 'Kannada', 'Kazakh', 'Korean', 'Latvian',
                        'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malay', 'Maori', 'Marathi',
                        'Mongolian', 'Norwegian', 'Persian', 'Polish', 'Portuguese', 'Portuguese - Brazil',
                        'Portuguese - Portugal', 'Punjabi (Gurmukhi)', 'Romanian', 'Russian',
                        'Serbian', 'Simplified Chinese', 'Slovak', 'Slovenian', 'Spanish - Latin America',
                        'Spanish - Spain', 'Swahili', 'Swedish', 'Tamil', 'Telugu', 'Thai',
                        'Traditional Chinese', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Valencian',
                        'Vietnamese', 'Welsh'] # List of languages, gotten by manually cleaning unique values of languages
                                                # Yes.. apparently Portuguese has 3 derivatives that are unique
                                                # Apparently this is true for a few of these languages. Nice to know!

    # The languages feature is turning out to be really... terrible in format
    # Tons of random \r\n tags, random BBCode and HTML tags, random parenthesis tags, we need to a ton of cleaning to extract and explode this column.
    def clean_languages(language_text):
        if pd.isnull(language_text) or language_text.strip() == '':
            return []
        language_text = language_text.replace('\r\n', ',').replace(';', ',')# Replace '\r\n' and ';' with commas
        language_text = re.sub(r'\[.*?\]', '', language_text) # Remove BBCode or HTML tags like '[b]*[/b]'
        language_text = language_text.replace('(full audio)', '').replace('Not supported', '') # Remove '(full audio)', 'Not supported'
        language_text = re.sub(' +', ' ', language_text) # Fix badly spaced language texts
        language_text = language_text.strip()

        if language_text == '':
            return []
        language_list = language_text.split(',') # Finally ready to split

        language_list = [lang.strip() for lang in language_list if lang.strip()]
        detected_languages = []
        for lang in language_list:
            matched = False
            for possible_lang in possible_languages:
                if possible_lang.lower() == lang.lower():
                    detected_languages.append(possible_lang)
                    matched = True
                    break
            if not matched:
                for possible_lang in possible_languages:
                    if possible_lang.lower() in lang.lower():
                        detected_languages.append(possible_lang)
                        break
        detected_languages = list(set(detected_languages))
        return detected_languages


    all_55000_steam['languages_cleaned'] = all_55000_steam['languages'].apply(clean_languages)
    mlb = MultiLabelBinarizer()

    languages_encoded = mlb.fit_transform(all_55000_steam['languages_cleaned'])
    languages_df = pd.DataFrame(languages_encoded, columns=mlb.classes_, index=all_55000_steam.index)

    all_55000_steam = pd.concat([all_55000_steam, languages_df], axis=1)
    all_55000_steam = all_55000_steam.drop('languages_cleaned', axis=1)
    all_55000_steam['languages_supported'] = languages_df.sum(axis=1)
    return all_55000_steam

modeling_copy = all_55000_steam.copy()
modeling_copy = perform_additional_feature_engineering(modeling_copy)
columns = list(modeling_copy.columns)

idx_empty_string = columns.index('') # We have an empty string column for genre since some games don't have a genre attached
idx_price_category_start = columns.index('price_category_Free to Play')
idx_language_start = columns.index('Afrikaans')
idx_languages_supported = columns.index('languages_supported')


genres_columns = columns[idx_empty_string:idx_price_category_start]
price_category_columns = columns[idx_price_category_start:idx_language_start]
languages_columns = columns[idx_language_start:idx_languages_supported]

# Additional features (confounders)
additional_features = ['developer_experience', 'publisher_experience', 'languages_supported']
feature_columns = additional_features + price_category_columns + genres_columns + languages_columns
# Target
target_column = 'total_review_count'

data = modeling_copy[feature_columns + [target_column]]
print(data.shape)
print(data.columns.values)

# Separate features and target
X = data[feature_columns]
y = data[target_column]

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

# Linear Regression Model ------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
# print("Linear Regression MSE:", mse_lr)
# print("Linear Regression R^2:", r2_lr)

# Random Forest Regressor Model ------------------------------
rf_model = RandomForestRegressor(random_state=1111)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# print("Random Forest MSE:", mse_rf)
# print("Random Forest R^2:", r2_rf)

# XGBoost Regressor Model ------------------------------------
xgb_model = xgb.XGBRegressor(random_state=1111)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# print("XGBoost MSE:", mse_xgb)
# print("XGBoost R^2:", r2_xgb)

# Random Forest feature importance
importances_rf = rf_model.feature_importances_
feature_importance_rf = pd.Series(importances_rf, index=feature_columns).sort_values(ascending=False)

# print("Random Forest Feature Importances:")
# print(feature_importance_rf.head(10))

# XGBoost feature importance
importances_xgb = xgb_model.feature_importances_
feature_importance_xgb = pd.Series(importances_xgb, index=feature_columns).sort_values(ascending=False)

# print("XGBoost Feature Importances:")
# print(feature_importance_xgb.head(10))

status_message = "Model validation complete! ✔️"

st.write(
    """
    Results:
    """
)

# Metrics
metrics_data = {
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "MSE": [mse_lr, mse_rf, mse_xgb],
    "R²": [r2_lr, r2_rf, r2_xgb]
}

metrics_df = pd.DataFrame(metrics_data)

# # Get coefficients
# coefficients = lr_model.coef_

# # Map coefficients to feature names
# coefficients_df = pd.DataFrame({
#     "Feature": X.columns,
#     "Coefficient": coefficients
# })

# # Extract coefficient for "developer_experience"
# developer_experience_coef = coefficients_df.loc[coefficients_df["Feature"] == "developer_experience", "Coefficient"].values[0]

# # Display result
# st.write(f"Coefficient for developer_experience: {developer_experience_coef}")

st.subheader("Model Performance Metrics")
st.table(metrics_df)

# Feature Importances
# Random Forest
st.subheader("Random Forest Feature Importances")
st.table(feature_importance_rf.head(10))

# XGBoost
st.subheader("XGBoost Feature Importances")
st.table(feature_importance_xgb.head(10))

st.write(
    """
    XGBoost out performs the rest of the models in terms of variance explained.
    - Linear Regression performs the worst, suggesting that the relationship between the predictors and the target variable is not strictly linear.
    - Random Forest improves upon Linear Regression, indicating that non-linear relationships and interactions between features are better captured.
    - XGBoost further enhances performance by effectively modeling complex patterns through gradient boosting.

    Dominant Feature - 🌟Developer Experience🌟:
    - Both models highlight "developer_experience" as the most influential feature! Though with XGBoost, we should note caveats in interpretability of important features, 
    we can underscore that indeed, developer experience has the most significant effect on game success out of the features we tested! [[18](https://arxiv.org/abs/1801.04293))
    """
)
        
    # with st.container(border=True):
    #     st.write(
    #         """
    #         From our qualitative analysis, we gathered these key insights:

    #         #### Regarding reviews: 
    #         - There seems to be "dominators" in this distribution. A small number of top-performing games account for the majority of reviews.
    #             - This reflects the industry's competitive nature, where a few titles capture most of the attention and player engagement due to factors like higher budgets, marketing, and established fan bases.
            
    #         - Suprisingly, over 9% of games have accumulated more than 1000 reviews, an unexpectedly high figure.
    #             - This could be due to social sharing, niche communities, or successful post-launch updates that maintain or grow player engagement.
    #             - This indicates a longer tail of the market, some creators are dominators, but there is a long tail of successful games that go unrecognized.
    #             - 🌟Niche communities or successful post-launch updates could be contributing to this long tail.🌟

    #         - A higher "like ratio" tends to correspond with larger player bases.
    #             - Positive reception often signals quality or enjoyment, attracting more players through word of mouth and organic growth.
            
    #         #### Regarding release date:
            
    #         #### Regarding genres:
    #         - Our most popular genres seem to be Action, Indie, and Casual.
    #             - These genres often offer broad appeal, allowing for a variety of gameplay styles and experiences that attract diverse audiences.
            
    #         - Despite the popularity of certain genres, success in terms of revenue or engagement isn't guaranteed.
    #             - Success might depend more on execution, marketing, and innovation rather than the genre itself, showing that not all popular genres translate into high-performing games.
    #         """
    #     )

    #     st.write(
    #         """
    #         Explore genre trends in this dataset with this interactive plot:
    #         """
    #     )
        