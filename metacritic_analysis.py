import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_metacritic, literal_evaluate

import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
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
2. [Feature Exploration](#feature-exploration)
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
    Before getting into the analysis, let's first explain our key features and how we intend to utilize them. We must also explain some various caveats regarding the 
    integrity of some particular features.

    In this metacritic dataset, we have a somewhat limited featureset, save for one remarkable feature: "platform_info", which will be the focus of our analysis for this page. By analyzing
    a dataset that is not Steam, we widen our perspective for a more integral understanding of trends. Steam may have some particular biases within itself, having an outside perspective allows 
    us to potentially uncover these oddities.
        
    - `platform_info`:
        - This particular feature has information regarding metacritic 'critic' reviews for different platforms of a game. With this, we can finally draw comparison between critics, and users.
        We can examine whether users prefer different games over critics, and whether critical acclaim is correlated to success.
        - In addition, we can address some ideas about the challenges of console releases. Games are often released under many different platforms at once to potentially capture a larger audience.
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

st.write(
    """
    We have a 0.49 value correlation between platform_metascore and user_score. 
    
    Combining numerical statistics on user score (min 3, med 71, max 100, avg 68.7, stddev 13.76) and platform_metascore (min 11, med 73, max 99, avg 71.08, stddev 12.1), we can surmise that while 
    users and critics generally agree in the same direction, their magnitudes may not be the same.
    
    Additionally, some will say critics [[5](https://www.gamesradar.com/its-time-for-metacritic-to-stand-up-to-review-bombings/)] typically take a more 'fair' / adjusted view to the games, players
    can sometimes deviate greatly from this. Potential effects from 'review bombing/boosting' can help explain this deviation.
    """
)

# ------------------------------------

st.subheader("Platforms Info Analysis", divider=True)

st.write(
    """
        Within this branch, we want to investigate the following questions:
        - What does our platform distribution look like?
        - Are users similar to critic review? Why might they be different?
        - What genres of games are most popular? What genres of games are most highly rated?
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
        Here we have plotted our distribution of platforms for our dataset.
        - Keep in mind, games can be part of multiple different platforms. After exploding our platform_info feature, we noticed that we had nearly triple the number of observations.
        - Games on different platforms can be under the same name, but function almost entirely different. As mentioned before, things like hardware limitations, and changes in control scheme.


        Although PC is the most dominant single game platform, the combined total from consoles constitutes a larger total body of published games.
        - Consoles offer a removal of barriers for entry to a lot of players [[6](https://www.psu.com/news/the-evolution-of-gaming-from-traditional-consoles-to-the-rise-of-online-platforms/)]. In recent years,
        console power to price has been becoming more accessible to many.
        - PC gaming's current high proportion may be from the current rise in indie games, since indie games are normally accessible even for weaker PCs (a recent phenonmenon)[[7](https://medium.com/@techgamernexus/the-rise-of-indie-games-how-small-studios-are-making-big-waves-46f6c495bf42#:~:text=The%20indie%20game%20revolution%20began%20in%20earnest%20in,traditional%20gatekeepers%20and%20reach%20a%20global%20audience%20directly.)].
    
        What may obfuscate the true nature of these observations is the idea that lots of games released on PC may also be released on console (and vice versa).
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
        These boxplots showcase metacritic scores for our top 5 platforms in popularity.
        
        While PC and others have similar rating distributions, iOS games seem to take a higher average rating!
        - This may be due to a difference in reviewers, its possible that different platforms have specifically different reviewers. PC, PS4, Xbox and Switch titles are often quite similar to one another, very
        often sharing the same game titles for release. The mobile game market, due to high differences in controls and accessibility, is much more distinct from the pack. While many critics will review across platforms,
        its possible that mobile game reviewers specialize.
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
        Looking at our distribution for critic review scores, we can see a relatively normal distribution, ranging mostly from 60 - 90 with a slight skew to the left.
        - Again, we see a growing idea: critics tend to give more moderate scores.
            - Critic reviewers have an ethical task to rate games that are similar in quality similarily [[8](https://cogconnected.com/feature/videogame-review-scores/)]. In addition,
            they need to be mindful of their reputation; should a review go unbalanced, it may become a controversy amongst other reviews and user opinion. We speculate there are many
            factors at play which cause reviewers to maintain a more conservative rating system.
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
        Looking at our most popular (most reviewed) PC games with critics and users, we can notice though the lists coincide for some names (Baldur's Gate 3 and Cyberpunk 2077), there 
        are mostly differences.
        - 🌟In addition, we can notice that _every_ game on _both_ lists are AAA games🌟.
            - Its possible this is due to the marketing and visibility budget differences, AAA publishers typically have high budget productions with massive marketing campaigns to garner attention before releases.
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
        For our most popular games with all consoles, our top games for critics consist of _all console games_, while our top for users is an almost even split between console and PC.
        - Metascores for reviewed games by critics seem to heavily favor AAA console releases. Every game on this list is from a big time publisher, and on console.
        - User review counts seem biased toward particularly controversial games.
            - "The Last of Us Part II" was an infamously controversial game due to player opinion regarding the directing of the story for this sequel. This was, at the time of release, headline news.
            - "Cyberpunk 2077" infamously was plagued with performance, optimzation and worst of all major game breaking bugs upon release. The game had been in development for over a decade, the hype generated for release
            was immense, which only made the problems amplified by users.
            - "Warcraft III: Reforged" was a game that was heavily criticized for 'underdelivering' on its promises. It is a remake of the famous and critically acclaimed "Warcraft III: Frozen Throne" game from Blizzard. Because this
            remake discontinued the older release and itself was a very underwhelming experience, fans were quick to unleash judgement. This game is one example of "review bombing".
    """
)

st.pyplot(fig_2)

st.write(
    """
        By plotting the metascores for these popular games, we can qualify what popularity might mean in terms of rating.
        - From our user side, we can see that the most popular games are not neccessarily highly rated. "Warcraft III: Reforged", being a victim of 'review bombing' by users, is one such controversy.
        - Within our critics side, we we can note the most popular also seem to have lots of fluctuation. When compared to the average, only a few of these games are standouts.
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
        Here we have plotted our top 20 PC games (highest rated by critics) that have at least 20 different critic reviews. We plot both the critic score and user score for comparison.

        Generally, critic scores appear to be within 10 points of the user score, with user scores being _lower_ for all 20 top critically acclaimed games.
        - This could indicate that satisfying users is more difficult than satisfying critics. Potentially this observation can be used to redouble marketing and testing focus to some more grassroots or user-based approaches. 
            - One method of doing this that has recently gained popularity is through the process of ‘early-access’ – allowing users to purchase access to a game before it is officially released and play through limited 
            sections and provide feedback before the final product goes out.
            - Larian Studios (publishers for Baldur's Gate and Divinity Origin Sin series) have garnered a lot of attention for heavily utilizing this tactic in development, iterating on user feedback as a focus [[8](https://gdcvault.com/play/1026452/The-Making-of-Divinity-Original)].
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
        Looking at genre, we plot our top 20 most popular genres for critics and users.
        - Critic reviews potentially reveal what genres may have the most games come out for (since there are a mostly fixed/limited number of major critics).
        - User reviews potentially are an indicator of which genres currently have the biggest player bases.
        - We can see that there is some discrepancy between the lower half between critic and user, but the top genres (action rpg, FPS, action adventure) are roughly the same. They get the most coverage and are the most popular by userbase.
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
        Here we have plotted boxplots for our top rated genres (by median rating). To qualify the boxplots, we also plot frequency bars on another axis.
        - We can notice the IQR range these genres varies quite a bit, but more especially on the top user rated genres side.
            - Having a smaller IQR range within our top games gives meaning to how players or critics appreciate that genre. It may be that, because the genre is less frequent, the games that are recieved in the genre are
            better appreciated, since they are few and far between.
        - Most of the critic top rated genres have a pretty good frequency representation with the exception of (racing sim, space sim, baseball).
        - Top user rated genres have ones that are low frequency representation. Users more likely to give the few existing games in their genre a higher rating when there is less competition.
    """
)

st.pyplot(fig_2)

st.write(
    """
        Here, we plot the tail end, the bottom rated genres (by median rating) with frequency bars.
        - All of the genres represented here are really niche, with very few games represented in each genre on average (most around 50)
        
        It may be fruitful to do further analysis on these particular "failing genres" for:
        - What kinds of studios make these games from less represented/lower rated genres. Are they shovelware/ AA, indie/AAA?
        - What users are saying about these types of games in these genres?
        - How many copies have these games sold? Are they actually popular/worth considering?

        As of now, however, we do not have enough information to speculate on these questions.
    """
)

# ------------------------------------

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
            🌟User and Critics Disagree🌟
            """
        )

        
        @st.fragment()
        def plot_most_critically_acclaimed_pc_games():
            at_least_n_reviews = st.slider("Set a number of reviews threshold:", min_value=0, max_value=50, value=20)
            pc_games = metacritic[metacritic['platform_name'] == 'PC']
            critically_acclaimed_pc_games = pc_games[pc_games['platform_metascore_count'] > at_least_n_reviews]
            critically_acclaimed_pc_games = critically_acclaimed_pc_games.sort_values(by='platform_metascore', ascending=False)

            top_20_games = critically_acclaimed_pc_games.head(20)

            colors = pc.qualitative.Plotly

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=top_20_games['platform_metascore'],
                y=top_20_games['name'],
                orientation='h',
                name='Critic metascore',
                marker=dict(color=colors[0])
            ))

            fig.add_trace(go.Bar(
                x=top_20_games['user_score'],
                y=top_20_games['name'],
                orientation='h',
                name='User score',
                marker=dict(color=colors[1])
            ))

            fig.update_layout(
                title=f'Top 20 PC games by rating (with more than {at_least_n_reviews} critic reviews) vs user score',
                legend=dict(x=.8, y=1),
                xaxis=dict(title='Score', range=[70, 100]),
                yaxis=dict(title='Game title'),
                barmode='overlay',
                height=600,
                width=900
            )
            st.plotly_chart(fig)
        plot_most_critically_acclaimed_pc_games()

        st.write(
            """
            - Critics tend to be within the 60 - 90 range for ratings. They seem to be fairer in assessment, at least in the similar platforms.
            - Users tend to rate more variably, for this list of top 20 critically acclaimed games, users on average rate these about 10 points less.
                - Users and critics do tend to rate in a similar direction though! 
            """
        )

    with st.container(border=True):
        st.write(
            """
            🌟Critic Favors in Genre🌟
            """
        )

        @st.fragment()
        def plot_top_and_bottom_rated_genres_and_frequencies():
            tail = st.toggle("Plot bottom instead of top")
            n = st.slider("Set a frequency threshold:", min_value=0, max_value=1000, value=0)
            genre_frequencies = metacritic['genres'].value_counts()
            genre_frequencies_df = pd.DataFrame({
                'genres': genre_frequencies.index,
                'count': genre_frequencies.values
            })
            genre_frequencies_df = genre_frequencies_df[genre_frequencies_df['count'] >= n]
            critics_genre_average_score = metacritic.groupby('genres')['platform_metascore'].agg('median')
            critics_genre_average_score_df = pd.DataFrame({
                'genres': critics_genre_average_score.index,
                'average_critic_review_score': critics_genre_average_score.values
            })
            critics_genre_average_score_df = critics_genre_average_score_df.sort_values(by='average_critic_review_score', ascending=False)
            top_20_genres_by_critic_average_score = critics_genre_average_score_df[critics_genre_average_score_df['genres'].isin(genre_frequencies_df['genres'])].head(20)['genres']
            filtered_metacritic_by_top_20_genres = metacritic[metacritic['genres'].isin(top_20_genres_by_critic_average_score)]

            if tail:
                top_20_genres_by_critic_average_score = critics_genre_average_score_df[critics_genre_average_score_df['genres'].isin(genre_frequencies_df['genres'])].tail(20)['genres']
                filtered_metacritic_by_top_20_genres = metacritic[metacritic['genres'].isin(top_20_genres_by_critic_average_score)]
            
            colors = pc.qualitative.Plotly

            # Boxes
            fig = go.Figure()
            for genre in top_20_genres_by_critic_average_score:
                genre_data = filtered_metacritic_by_top_20_genres[filtered_metacritic_by_top_20_genres['genres'] == genre]
                fig.add_trace(go.Box(
                    x=genre_data['platform_metascore'],
                    name=genre,
                    orientation='h',
                    boxmean=True,
                    marker=dict(color=colors[1]),
                    xaxis='x'
                ))

            genre_frequencies_df = genre_frequencies_df.set_index('genres').reindex(top_20_genres_by_critic_average_score)

            # Frequency bars
            fig.add_trace(go.Bar(
                y=top_20_genres_by_critic_average_score,
                x=genre_frequencies_df['count'],
                orientation='h',
                marker=dict(color='rgba(128, 128, 128, 0.3)'),
                xaxis='x2'
            ))

            fig.update_layout(
                title="Top rated genres by critics (with frequency)",
                xaxis=dict(
                    title="Metascore",
                    side='bottom'
                ),
                xaxis2=dict(
                    title="Count",
                    side='top',
                    overlaying='x'
                ),
                yaxis=dict(
                    title="Genre"
                ),
                barmode='overlay',
                height=600,
                width=900,
                showlegend=False
            )
            if tail:
                fig.update_layout(title="Bottom rated genres by critics (with frequency)")

            st.plotly_chart(fig)

        plot_top_and_bottom_rated_genres_and_frequencies()
        
        st.write(
            """
            - Certain genre with smaller IQR maybe more appreciated by critics, for example racing games are very uncommon, when they do
            show, they may be looked on with more favor.
            - Genre with higher frequency and also smaller IQR can be considered having some integral favor with critics. Metroidvania seems to be a 
            rather popular genre with also a very heavy favor with critics.
                - There are an immense amount of heavily critically acclaimed metroidvania titles [[9](https://www.denofgeek.com/games/the-best-metroidvania-games-ever/)][[10](https://www.gamesradar.com/best-metroidvania-games/)].
                - For niche genres, it's often a handful of standout titles that establish the benchmark for quality. Perhaps these benchmarks cause games that follow
                to meet critic expectations well.
            """    
        )