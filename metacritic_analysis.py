import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_metacritic, literal_evaluate

import plotly.express as px
import plotly.graph_objects as go
import re

metacritic = load_metacritic()
plt.style.use('ggplot')

def perform_feature_engineering(metacritic: pd.DataFrame) -> pd.DataFrame:
    # Standardizing the metacritic platforms info data
    metacritic['platforms_info'] = metacritic['platforms_info'].apply(literal_evaluate)

    def extract_metascore_count(text):
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        return None

    metacritic_exploded_platforms = metacritic.explode('platforms_info')
    metacritic_exploded_platforms.rename(mapper={'platforms_info': 'platform_info'}, axis=1, inplace=True)
    new_feature_columns = []
    for index, row in metacritic_exploded_platforms.iterrows():
        platform_info = row['platform_info']
        if type(platform_info) is not dict:
            new_feature_columns.append({
                'platform_name': np.nan,
                'platform_metascore': np.nan,
                'platform_metascore_count': np.nan
            })
            continue
        platform_name = platform_info.get('Platform')
        platform_metascore = platform_info.get('Platform Metascore')
        platform_metascore_count_text = platform_info.get('Platform Metascore Count')
        platform_metascore_count = extract_metascore_count(platform_metascore_count_text)
        new_feature_columns.append({
            'platform_name': platform_name,
            'platform_metascore': platform_metascore,
            'platform_metascore_count': platform_metascore_count
        })
    new_feature_columns = pd.DataFrame(new_feature_columns)
    new_feature_columns['platform_metascore'] = pd.to_numeric(new_feature_columns['platform_metascore'], errors='coerce').astype("Int64")
    new_feature_columns['platform_metascore_count'] = pd.to_numeric(new_feature_columns['platform_metascore_count'], errors='coerce').astype("Int64")
    metacritic_exploded_platforms.reset_index(drop=True, inplace=True)

    metacritic = pd.concat([metacritic_exploded_platforms, new_feature_columns], axis=1)

    metacritic['release_date'] = pd.to_datetime(metacritic['release_date'], format='%Y-%m-%d')
    return metacritic
metacritic = perform_feature_engineering(metacritic)

st.header("🕹️Exploring Metacritic Reviews🕹️")

st.write(
    """
    In this page, we will explore the metacritic dataset. The dataset contains all the video games that have been featured on Metacritic.com from 1995 to January 2024, 
    around 16 thousand entries across all platforms and genres.
    """
)

st.markdown("""
1. [Features of Interest](#features-of-interest)
    * A preface on the unique features for this particular dataset.
2. [Feature Explorations](#feature-explorations)
    * The exploration journey used to uncover key insights. Jumping to our dashboard is an option, but we _encourage_ a review of our exploration to really grasp understanding
    of our findings.
    * Because this dataset has only one key feature we focus on, we are not branching at this time.
3. [Key Insights](#key-insights)
    * A comprehensive list of our most important findings, along with some caveats to consider.
""")

# Insert dashboards:
st.subheader("Features of Interest", divider=True)
st.write(
    """
    Before getting into the analysis, let's first explain our key features and how we intend to utilize them. We must also explain some various caveots regarding the 
    integrity of some particular features.

    In this metacritic dataset, we have a somewhat limited featureset, save for one remarkable feature: "platform_info", which will be the focus of our analysis for this page. By analyzing
    a dataset that is not Steam, we widen our perspective for a more integral understanding of trends. Steam may have some particular biases within itself, having an outside perspective allows 
    us to potentially uncover these oddities.
        
    - `platform_info`:
        - This particular feature has information regarding metacritic 'critic' reviews for different platforms of a game. With this, we can finally draw comparison between critics, and users.
        We can examine whether users perfer different games over critics, and whether critical acclaim is correlated to success.
        - In addition, we can address some ideas about the challenges of console releases. Games are often released under many different platforms at once to potentially capture hidden audiences.
        With review score information per platform, we can address how different console challenges affect game success. We may also investigate whether certain consoles are more suited for certain 
        genre.
            - Different platform releases each offer a unique challenge for developers. Platforms have different control limitations, compatability isuses, hardware, etc.
    """
)

st.write("See our documentation page for further details on this dataset's features set:")

st.page_link("./documentation.py", label="Documentation", icon="📔")

st.subheader("Feature Exploration", divider=True)

st.write("#### General items")
st.write(
    """
        Basic statistics with feature listings:
    """
)

numerical_stats_tab, object_stats_tab = st.tabs(["Numerical Statistics", "Categorical Statistics"])

with numerical_stats_tab:
    st.write("### Numerical feature statistics")
    st.write(metacritic.describe())

with object_stats_tab:
    st.write("### Categorical feature statistics")
    st.write(metacritic[metacritic.select_dtypes(include=['object', 'category']).columns].drop('platform_info', axis=1).describe())

def plot_correlation_heatmap() -> go.Figure:
    corr_matrix = metacritic.select_dtypes(include='number').corr()
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
        margin=dict(l=150, r=120, t=100, b=150)
    )
    
    return fig

st.plotly_chart(plot_correlation_heatmap())

# ------------------------------------

st.subheader("Platforms Info Analysis", divider=True)

st.write(
    """
        Within this branch, we want to investigate the following questions:
        - TODO: ADD EXPLORATION QUESTIONS
    """
)

def plot_platform_frequencies() -> plt.Figure:  
    platform_frequencies = metacritic.groupby('platform_name').size()
    platform_frequencies.name = 'count'
    platform_frequencies = pd.DataFrame({
        'platform_name': platform_frequencies.index,
        'count': platform_frequencies.values
    })
    platform_frequencies.sort_values(by='count', ascending=True, inplace=True)

    fig = plt.figure(figsize=(12,6))
    ax = sns.barplot(
        data=platform_frequencies,
        x='count',
        y='platform_name',
        orient='h',
    )
    ax.bar_label(ax.containers[0])
    plt.title('Platform frequencies')
    plt.xlabel('Count')
    plt.ylabel('Platform')
    return fig

st.pyplot(plot_platform_frequencies())

st.write(
    """
        TODO: COMMENT
    """
)

def plot_most_popular_platforms_boxplots() -> plt.Figure:
    most_popular_platforms = ['PC', 'PlayStation 4', 'Nintendo Switch', 'Xbox One', 'iOS (iPhone/iPad)']
    most_popular_platforms = metacritic.loc[metacritic['platform_name'].isin(most_popular_platforms)]

    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(
        data = most_popular_platforms,
        x = 'platform_metascore',
        y = 'platform_name',
        orient='h'
    )
    plt.title("Most popular platforms against critics review score")
    plt.xlabel("Metacritic score")
    plt.ylabel("Platform")
    return fig
st.pyplot(plot_most_popular_platforms_boxplots())

st.write(
    """
        TODO: COMMENT
    """
)

pc_games = metacritic[metacritic['platform_name'] == 'PC']
ascending = False
n = 10
most_popular_pc_games_by_critics = pc_games.sort_values(by='platform_metascore_count', ascending = ascending)
most_popular_pc_games_by_users = pc_games.sort_values(by='user_ratings_count', ascending = ascending)

def plot_critic_review_count_distribution() -> plt.Figure:
    fig = plt.figure(figsize=(12, 6))
    sns.histplot(
        data = metacritic,
        x = 'platform_metascore',
        bins = 20,
    )
    plt.title("Critics review scores distribution")
    plt.xlabel("Metascore")
    plt.ylabel("Count")

    return fig
st.pyplot(plot_critic_review_count_distribution())

st.write(
    """
        TODO: COMMENT
    """
)

def plot_most_popular_pc_games_by_critics_and_users(n = 10) -> plt.Figure:
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes: list[plt.Axes] = axes.ravel()
    sns.barplot(
        data = most_popular_pc_games_by_critics.head(n),
        x = 'platform_metascore_count',
        y = 'name',
        ax = axes[0]
    )
    axes[0].set_xlim([80, 120])
    axes[0].set_title('Critic highlights: PC games with most metacritic reviews')
    axes[0].set_xlabel('Review count')
    axes[0].set_ylabel('Game title')
    
    sns.barplot(
        data = most_popular_pc_games_by_users.head(n),
        x = 'user_ratings_count',
        y = 'name',
        ax = axes[1]
    )
    axes[1].set_title('User highlights: PC games with most user reviews')
    axes[1].set_xlabel('Review count')
    axes[1].set_ylabel('Game title')

    plt.tight_layout()
    return fig
st.pyplot(plot_most_popular_pc_games_by_critics_and_users())

st.write(
    """
        TODO: COMMENT
    """
)

def plot_most_popular_platforms() -> plt.Figure:
    metacritic['name_and_platform'] = metacritic['name'] + ' (' + metacritic['platform_name'] + ')'
    most_popular_by_critics = metacritic.sort_values(by='platform_metascore_count', ascending = ascending)
    most_popular_by_users = metacritic.sort_values(by='user_ratings_count', ascending = ascending)

    fig_1, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes: list[plt.Axes] = axes.ravel()
    sns.barplot(
        data = most_popular_by_critics.head(n),
        x = 'platform_metascore_count',
        y = 'name',
        hue = 'platform_name',
        ax = axes[0]
    )
    axes[0].set_title('Critic highlights: most popular with critics across platforms')
    axes[0].set_xlabel('Review count')
    axes[0].set_ylabel('Game title')
    axes[0].set_xlim([120, 160])
    sns.barplot(
        data = most_popular_by_users.head(n),
        x = 'user_ratings_count',
        y = 'name_and_platform',
        hue = 'platform_name',
        ax = axes[1]
    )
    axes[1].set_title('User highlights: most popular with users across platforms')
    axes[1].set_xlabel('Review count')
    axes[1].set_ylabel('Game title')

    plt.tight_layout()

    fig_2, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(
        data = most_popular_by_critics.head(n),
        x = 'platform_metascore',
        y = 'name',
        hue = 'platform_name',
        ax = axes[0]
    )
    axes[0].set_title('Metascores for the games with the most critic reviews')
    axes[0].set_xlabel("Metascore")
    axes[0].set_ylabel('Game title')
    axes[0].set_xlim([60, 100])

    sns.barplot(
        data = most_popular_by_users.head(n),
        x = 'user_score',
        y = 'name_and_platform',
        hue = 'platform_name',
        ax = axes[1]
    )
    axes[1].set_title('Metascores for the games with the most user reviews')
    axes[1].set_xlabel("Metascore")
    axes[1].set_ylabel('Game title')

    plt.tight_layout()
    return fig_1, fig_2
fig_1, fig_2 = plot_most_popular_platforms()
st.pyplot(fig_1)

st.write(
    """
        TODO: COMMENT
    """
)

st.pyplot(fig_2)

st.write(
    """
        TODO: COMMENT
    """
)

def plot_most_critically_acclaimed_pc_games(at_least_n_reviews) -> plt.Figure:
    critically_acclaimed_pc_games = pc_games[pc_games['platform_metascore_count'] > at_least_n_reviews]
    critically_acclaimed_pc_games = critically_acclaimed_pc_games.sort_values(by='platform_metascore', ascending=False)

    fig, ax = plt.subplots(figsize=(15, 6))

    sns.barplot(
        data=critically_acclaimed_pc_games.head(20),
        x='platform_metascore',
        y='name',
        label='Critic metascore',
        ax=ax
    )
    
    sns.barplot(
        data=critically_acclaimed_pc_games.head(20),
        x='user_score',
        y='name',
        label='User score',
        alpha=0.5,
        ax=ax,
    )

    ax.set_xlim([70, 100])
    ax.set_title(f'Top 20 PC games by rating that are critically acclaimed (more than {at_least_n_reviews} critic reviews) vs user score')
    ax.set_xlabel('Metascore')
    ax.set_ylabel('Game title')
    ax.legend(loc='lower right')

    plt.tight_layout()
    return fig
st.pyplot(plot_most_critically_acclaimed_pc_games(20))

st.write(
    """
        TODO: COMMENT
    """
)

def plot_genre_popularities() -> plt.Figure:
    critics_genre_popularity = metacritic.groupby('genres')['platform_metascore_count'].agg('sum')
    critics_genre_popularity = pd.DataFrame({
        'genre': critics_genre_popularity.index,
        'critic_review_count': critics_genre_popularity.values
    })
    critics_genre_popularity = critics_genre_popularity.sort_values(by='critic_review_count', ascending=False)

    users_genre_popularity = metacritic.groupby('genres')['user_ratings_count'].agg('sum')
    users_genre_popularity = pd.DataFrame({
        'genre': users_genre_popularity.index,
        'user_rating_count': users_genre_popularity.values
    })

    users_genre_popularity = users_genre_popularity.sort_values(by='user_rating_count', ascending=False)


    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(
        x=critics_genre_popularity.head(20)['critic_review_count'],
        y=critics_genre_popularity.head(20)['genre'],
        orient='h',
        ax=axes[0],
    )
    axes[0].set_title('Top 20 most popular genres by critic review counts')
    axes[0].set_xlabel('Review count')
    axes[0].set_ylabel('Genre')

    sns.barplot(
        x=users_genre_popularity.head(20)['user_rating_count'],
        y=users_genre_popularity.head(20)['genre'],
        orient='h',
        ax=axes[1],
    )
    axes[1].set_title('Top 20 most popular genres by user rating counts')
    axes[1].set_xlabel('Rating count')
    axes[1].set_ylabel('Genre')

    plt.tight_layout()
    return fig
st.pyplot(plot_genre_popularities())

st.write(
    """
        TODO: COMMENT
    """
)

def plot_top_and_bottom_rated_genres_and_frequencies() -> list[plt.Figure, plt.Figure]:
    # Top Rated --------------------------- 
    genre_frequencies = metacritic['genres'].value_counts()
    genre_frequencies = pd.DataFrame({
        'genres': genre_frequencies.index,
        'count': genre_frequencies.values
    })

    critics_genre_average_score = metacritic.groupby('genres')['platform_metascore'].agg('median')
    critics_genre_average_score = pd.DataFrame({
        'genres': critics_genre_average_score.index,
        'average_critic_review_score': critics_genre_average_score.values
    })
    critics_genre_average_score = critics_genre_average_score.sort_values(by='average_critic_review_score', ascending=False)
    top_20_genres_by_critic_average_score = critics_genre_average_score.head(20)['genres']
    filtered_metacritic_by_top_20_genres_by_critic_average_score = metacritic[metacritic['genres'].isin(top_20_genres_by_critic_average_score)]

    top_rated_fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    axes: list[plt.Axes] = axes.ravel()
    sns.boxplot(
        data=filtered_metacritic_by_top_20_genres_by_critic_average_score,
        x='platform_metascore',
        y='genres',
        orient='h',
        order=top_20_genres_by_critic_average_score,
        ax=axes[0]
    )
    axes[0].set_title("Top rated genres by critics (with frequency)")
    axes[0].set_xlabel("Metascore")
    axes[0].set_ylabel("Genre")

    # Pandas sort_values accepts a parameter "key", allowing us to sort columns with a function.
    # In this case, I want to sort the genre_frequencies by the order of the top 20 genres selected earlier, to match the values to correct genre when plotting the count bars.
    # This was highly informative: https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas/72256842#72256842
    def genres_sorter(column):
        sorted_list = top_20_genres_by_critic_average_score.values
        correspondence = {genre: order for order, genre in enumerate(sorted_list)}
        return column.map(correspondence)

    genre_frequencies = genre_frequencies.sort_values(
        by="genres",
        key=genres_sorter
    )

    frequency_bars_ax: plt.Axes = axes[0].twiny()
    frequency_bars_ax.barh(
        top_20_genres_by_critic_average_score,
        genre_frequencies[genre_frequencies['genres'].isin(top_20_genres_by_critic_average_score)]['count'],
        color="gray",
        alpha=0.3,
    )
    frequency_bars_ax.set_xlabel('Count')

    # ---------------------------

    users_genre_average_score = metacritic.groupby('genres')['user_score'].agg('median')
    users_genre_average_score = pd.DataFrame({
        'genres': users_genre_average_score.index,
        'average_user_rating_score': users_genre_average_score.values
    })

    users_genre_average_score = users_genre_average_score.sort_values(by='average_user_rating_score', ascending=False)
    top_20_genres_by_user_average_score = users_genre_average_score.head(20)['genres']
    filtered_metacritic_by_top_20_genres_by_user_average_score = metacritic[metacritic['genres'].isin(top_20_genres_by_user_average_score)]

    sns.boxplot(
        data=filtered_metacritic_by_top_20_genres_by_user_average_score,
        x='user_score',
        y='genres',
        orient='h',
        order=top_20_genres_by_user_average_score,
        ax=axes[1]
    )
    axes[1].set_title("Top rated genres by users (with frequency)")
    axes[1].set_xlabel("Metascore")
    axes[1].set_ylabel("Genre")

    def genres_sorter(column):
        sorted_list = top_20_genres_by_user_average_score.values
        correspondence = {genre: order for order, genre in enumerate(sorted_list)}
        return column.map(correspondence)

    genre_frequencies = genre_frequencies.sort_values(
        by="genres",
        key=genres_sorter
    )

    frequency_bars_ax: plt.Axes = axes[1].twiny()
    frequency_bars_ax.barh(
        top_20_genres_by_user_average_score,
        genre_frequencies[genre_frequencies['genres'].isin(top_20_genres_by_user_average_score)]['count'],
        color="gray",
        alpha=0.3,
    )
    frequency_bars_ax.set_xlabel('Count')
    plt.tight_layout()

    # Bottom Rated --------------------------- 

    bottom_20_genres_by_critic_average_score = critics_genre_average_score.tail(20)['genres']
    filtered_metacritic_by_bottom_20_genres_by_critic_average_score = metacritic[metacritic['genres'].isin(bottom_20_genres_by_critic_average_score)]

    bottom_rated_fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    axes: list[plt.Axes] = axes.ravel()
    sns.boxplot(
        data=filtered_metacritic_by_bottom_20_genres_by_critic_average_score,
        x='platform_metascore',
        y='genres',
        orient='h',
        order=bottom_20_genres_by_critic_average_score,
        ax=axes[0]
    )
    axes[0].set_title("Worst rated genres by critics (with frequency)")
    axes[0].set_xlabel("Metascore")
    axes[0].set_ylabel("Genre")

    # Pandas sort_values accepts a parameter "key", allowing us to sort columns with a function.
    # In this case, I want to sort the genre_frequencies by the order of the top 20 genres selected earlier, to match the values to correct genre when plotting the count bars.
    # This was highly informative: https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas/72256842#72256842
    def genres_sorter(column):
        sorted_list = bottom_20_genres_by_critic_average_score.values
        correspondence = {genre: order for order, genre in enumerate(sorted_list)}
        return column.map(correspondence)

    genre_frequencies = genre_frequencies.sort_values(
        by="genres",
        key=genres_sorter
    )

    frequency_bars_ax: plt.Axes = axes[0].twiny()
    frequency_bars_ax.barh(
        bottom_20_genres_by_critic_average_score,
        genre_frequencies[genre_frequencies['genres'].isin(bottom_20_genres_by_critic_average_score)]['count'],
        color="gray",
        alpha=0.3,
    )
    frequency_bars_ax.set_xlabel('Count')

    # ---------------------------

    bottom_20_genres_by_user_average_score = users_genre_average_score.tail(20)['genres']
    filtered_metacritic_by_bottom_20_genres_by_user_average_score = metacritic[metacritic['genres'].isin(bottom_20_genres_by_user_average_score)]

    sns.boxplot(
        data=filtered_metacritic_by_bottom_20_genres_by_user_average_score,
        x='user_score',
        y='genres',
        orient='h',
        order=bottom_20_genres_by_user_average_score,
        ax=axes[1]
    )
    axes[0].set_title("Worst rated genres by users (with frequency)")
    axes[0].set_xlabel("Metascore")
    axes[0].set_ylabel("Genre")

    def genres_sorter(column):
        sorted_list = bottom_20_genres_by_user_average_score.values
        correspondence = {genre: order for order, genre in enumerate(sorted_list)}
        return column.map(correspondence)

    genre_frequencies = genre_frequencies.sort_values(
        by="genres",
        key=genres_sorter
    )

    frequency_bars_ax: plt.Axes = axes[1].twiny()
    frequency_bars_ax.barh(
        bottom_20_genres_by_user_average_score,
        genre_frequencies[genre_frequencies['genres'].isin(bottom_20_genres_by_user_average_score)]['count'],
        color="gray",
        alpha=0.3,
    )
    frequency_bars_ax.set_xlabel('Count')
    plt.tight_layout()

    return top_rated_fig, bottom_rated_fig
fig_1, fig_2 = plot_top_and_bottom_rated_genres_and_frequencies()
st.pyplot(fig_1)

st.write(
    """
        TODO: COMMENT
    """
)

st.pyplot(fig_2)

st.write(
    """
        TODO: COMMENT
    """
)

# ------------------------------------

st.markdown("[Explore other branches](#branch-exploration)")

st.subheader("Key Insights", divider=True)