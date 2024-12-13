import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain
import nltk
from nltk import ngrams
from nltk.probability import FreqDist
import networkx as nx
import matplotlib.cm as cm
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud

import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc

from utils import load_steam_reviews
steam_reviews = load_steam_reviews()
plt.style.use('ggplot')

nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

st.header("💬Exploring Steam Reviews💬")

st.write(
    """
    In this page, we will explore the steam reviews dataset, a dataset containing around 21 million user reviews of different games on steam from this year up to 2021.
    The data was obtained via Steam's provided api, outlined in their official documentation [here](https://partner.steamgames.com/doc/store/getreviews).
    """
)

st.markdown("""
1. [Features of Interest](#features-of-interest)
    * A preface on the unique features for this particular dataset, and some caveats to watch out for.
2. [User Review Text Analysis](#user-review-text-analysis)
    * The exploration journey used to uncover key insights. Jumping to our dashboard is an option, but we _encourage_ a review of our exploration to really grasp understanding
    of our findings.
3. [Key Insights](#key-insights)
    * A comprehensive list of our most important findings.
""")

st.subheader("Features of Interest", divider=True)
st.write(
    """
    For this particular dataset, our main focus will be on the unique "text data" element. This dataset features a lot of different supplementary features that go 
    along with this key feature, but for our current analysis plans, we will be largely ignoring these other features. Here we highlight our main feature, and some
    particularly notable supplementary features:
        
    - `review`:
        - This feature consists of the raw text data in an individual user review on a particular game. For game developers, player feedback is of particular interest!
        Developers often experience a certain "expert-induced blindness" to their own game [[13](https://en.wikipedia.org/wiki/Curse_of_knowledge)][[14](https://daedtech.com/lifting-the-curse-of-knowledge/))]. 
        This phenomenon is highly documented; because developers often have deep knowledge of their game systems, they often have a hard time placing themselves in
        the players' persepective. 
        - Additionally, players and developers often prioritize different aspects of a game. Developers, as creators, tend to focus on elements like story direction, 
        gameplay mechanics, performance, and functionality. However, it is ultimately the players who serve as the true arbiters of the gaming experience, and they may
        be concerned with particular elements that the developer may not be as heavily focused on.
            - As mentioned in another analysis, some development teams place heavy emphasis on player feedback, placing heavy concern on player agency and a 
            dedicated feedback platform [[8](https://gdcvault.com/play/1026452/The-Making-of-Divinity-Original)].
    - `recommended`:
        - A boolean feature indicating whether the reviewer "recommends" the game to other users to try. Steam does not have a numerical rating system (e.g. stars out of 5)
        for reviewers, instead relying solely on this metric of recommended or not. We utilize this feature as a metric to separate reviews that are positive and negative.
        - Though we may gain key insights from examining reviews regardless of their polarity, this feature lets us investigate key insights into what drives user sentiment by
        acting as our ground truth.
    - `author_playtime_at_review`:
        - A numerical feature indicating the amount of time (in minutes) the reviewer had spent on the game at the time of their review submission.
        - A key question we may ask is whether more playtime correlates with a more positive review. This provides some valuable context on whether playtime should be something
        that a game developer should factor into their decision making.
            - In addition, we may utilize this feature to filter for outliers. Extremely low playtime coupled with a strong recommendation intuitively seems somewhat biased, 
            and may be a good candidate for a filtering out. On the flip side, players with extremely long playtime may also be subject to "expert blindness" and give feedback
            that is not necessarily aligned well with the average player.
    """
)

st.write(
    """
    ##### Some important caveats:
    1. Preprocessing and feature engineering for text data often involves a multitude of different steps and choices. There is often debate surrounding integrity issues when performing cleaning
    operations such as lemmatization, stemming, stopword removal etc. [[15](https://www.linkedin.com/advice/1/how-can-you-use-lemmatization-improve-your-text-data-wmpxe)]. For
    example, lemmatization groups together words that are similar in meaning, but this grouping also destroys certain nuances that similar words have unique to one another 
    (e.g. good, better, best all become "good"). For our cleaning steps, it should be noted that there are alternative methods that can be argued for and against. 
    Our choices made are largely for the sake of both ease of understanding, and uncovering integral general patterns.
    2. Due to space and speed limitations of working with such a large dataset, we have limited our analysis to a random selection of 300,000 english language reviews.
        - Furthermore, much of our analysis will only utilize a sample of these 300,000 reviews.
    """
)

code = '''
steam_reviews = steam_reviews.sample(sample_size)
'''
st.code(code, language='python')

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
    st.write(steam_reviews.describe())

with object_stats_tab:
    st.write("### Categorical feature statistics")
    st.write(
        steam_reviews[
            [
                "name",
                "language",
                "review",
                "recommended",
                "steam_purchase",
                "received_for_free",
                "written_during_early_access",
            ]
        ].describe()
    )

st.write("Numerical feature correlations")

def plot_correlation_heatmap() -> go.Figure:
    corr_matrix = steam_reviews.select_dtypes(include=['number', 'bool']).drop(['app_id', 'review_id', 'author_steamid'], axis=1).corr()
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
        This heatmap shows the correlation between different numerical features. For this particular dataset, besides the trivial correlations, we
        observe no great correlators with our engineered review features.
    """
)
# ------------------------------------

st.subheader("User Review Text Analysis", divider=True)

st.write(
    """
        Within this branch, we want to investigate the following questions:
        - What does our text data length distributions look like?
        - What are some common reviews words? Can we identify common words that carry meaning?
        - What are the most common complaints in reviews that have a negative recommendation?
        - Does playtime have any bearing on our recommendation? Does playtime have bearing on our review text shape?
            - Can we consider extremely short or extremely long playtime to be a good indicator of an outlier review?
    """
)

@st.cache_data
def plot_review_length_distributions() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.histplot(steam_reviews['review_text_length'], kde=True, bins=30, alpha=0.7, ax=axes[0])
    sns.histplot(steam_reviews['review_total_words'], kde=True, bins=30, alpha=0.7, ax=axes[1])
    sns.histplot(steam_reviews['review_total_sentences'], kde=True, bins=30, alpha=0.7, ax=axes[2])
    axes[0].set_xlabel("Length (by character)")
    axes[1].set_xlabel("Length (by word)")
    axes[2].set_xlabel("Length (by sentence)")

    return fig
st.pyplot(plot_review_length_distributions())

st.write(
    """
    Author text lengths are heavily skewed to the right, with a quick dropoff and long tail distribution. Evidently, most reviews are relatively short.
    - While the majority of users leave concise feedback, a few provide highly detailed and comprehensive reviews.
    - Looking at text length distribution can give us insight into common user behavioral patterns.

    Though all three distributions all showcase a similar metric, we can gleam different insight from each:
    - The distribution of review length by character gives us a very smooth curve. This tells us that our dropoff is rather consistent.
    - The distribution of review length by words gives us further information, most reviews contain 20 or fewer words, with there being slight peaks and valleys in this distribution.
    - The distribution of review length by sentence gives us our best metric in terms of conciseness and relatability. Most reviews are one or two sentences.
    
    To account for our heavy skew, we can plot only looking at lengths below a certain percentile.
    """
)

@st.cache_data
def plot_review_length_distributions_with_threshold(threshold_percentage: float = 0.10) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    threshold_percentage = 0.10
    rank_threshold = int(steam_reviews.__len__() * threshold_percentage)

    sorted_text_length = steam_reviews['review_text_length'].sort_values(ascending=False)
    sorted_word_count = steam_reviews['review_total_words'].sort_values(ascending=False)
    sorted_sentence_count = steam_reviews['review_total_sentences'].sort_values(ascending=False)

    filtered_text_length = sorted_text_length.iloc[rank_threshold:]
    filtered_word_count = sorted_word_count.iloc[rank_threshold:]
    filtered_sentence_count = sorted_sentence_count.iloc[rank_threshold:]

    sns.histplot(filtered_text_length, kde=True, bins=30, alpha=0.7, ax=axes[0])
    axes[0].set_title("Review #characters distribution (<= 90th percentile))", fontsize=10)

    sns.histplot(filtered_word_count, kde=True, bins=30, alpha=0.7, ax=axes[1])
    axes[1].set_title("Review #words distribution (<= 90th percentile))", fontsize=10)

    sns.histplot(filtered_sentence_count, discrete=True, alpha=0.7, ax=axes[2])
    axes[2].set_title("Review #sentences distribution (<= 90th percentile))", fontsize=10)

    axes[0].set_xlabel("Length (by character)")
    axes[1].set_xlabel("Length (by word)")
    axes[2].set_xlabel("Length (by sentence)")

    return fig
st.pyplot(plot_review_length_distributions_with_threshold())

st.write(
    """
    With these plots, we can qualify previous insights on the majority of users.
    - The vast majority of users (90%) leave responses that are quick and concise. Most leave reviews that are as short as 1 or 2 sentences.
    """
)

@st.cache_data
def plot_most_common_n_words(n: int = 20) -> plt.Figure:
    all_tokens = list(chain.from_iterable(steam_reviews['even_cleaner_tokenized_review']))
    freq_dist = FreqDist(all_tokens)

    n = 20
    top_n_words = freq_dist.most_common(n)
    top_n_words = pd.DataFrame(top_n_words, columns=['Word', 'Frequency'])
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data = top_n_words,
        x = 'Frequency',
        y = 'Word',
        orient = 'h'
    )
    plt.title(f"Top {n} most common words")
    return plt.gcf()
st.pyplot(plot_most_common_n_words())

st.write(
    """
    Here we plot our top 20 most common words:
    - Certain words like "game" and "play" dominate our most common words. And this is after stopword filtering!
    - Evidently, even with stopword filtering, we end up with tons of words which we still do not care too much about.

    Probably the most notable word on this list is "story".
    - A very hotly debated topic in the game development space involves whether "story" / narrative truly is a key aspect in marketability of games [17](https://www.gameskinny.com/culture/is-story-important-in-video-games-it-is-not-that-simple/)
    - It is interesting to note that story comes up so often! 🌟Story may be a key aspect in games that players most often look for 🌟. We may want to try to qualify this 
    statement later by looking at how keywords of relate to genre.
    """
)

def plot_top_n_bigram_frequencies(n = 20) -> plt.Figure:
    all_tokens = list(chain.from_iterable(steam_reviews['even_cleaner_tokenized_review']))

    n = 2
    n_grams = list(ngrams(all_tokens, n))
    freq_dist_ngrams = FreqDist(n_grams)

    top_n_ngrams = freq_dist_ngrams.most_common(20)

    # Convert to DataFrame for plotting
    top_n_ngrams_df = pd.DataFrame(top_n_ngrams, columns=['ngram', 'frequency'])
    top_n_ngrams_df['ngram'] = top_n_ngrams_df['ngram'].apply(lambda x: ' '.join(x))  # Convert tuples to strings

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_n_ngrams_df,
        x='frequency',
        y='ngram',
        orient='h'
    )
    plt.title(f'Top 20 most common {n}-grams')
    plt.xlabel('Frequency')
    plt.ylabel(f'{n}-gram')
    return plt.gcf()
st.pyplot(plot_top_n_bigram_frequencies())

st.write(
    """
    Looking at the most common bigrams we see most often, reviews associate adjectives with the general word "game". Using a word network graph...
    """
)

# Uses some borrowed functions from a previous ICA to plot a bigram word network
@st.fragment()
def plot_bigram_word_network() -> plt.Figure:
    if st.button("Redraw network graph"):
        st.rerun()

    def create_network_visualization(bigrams, min_freq=2, max_edges=50) -> nx.Graph:
        G = nx.Graph() # Initialize a graph object
        
        for (w1, w2), freq in bigrams.items(): # Add edges
            if freq >= min_freq:
                G.add_edge(w1, w2, weight=freq)
        
        if len(G.edges()) > max_edges: # Limit to most significant connections to avoid over cluttering
            significant_edges = sorted(G.edges(data=True), 
                                    key=lambda x: x[2]['weight'], 
                                    reverse=True)[:max_edges]
            G = nx.Graph()
            for w1, w2, data in significant_edges:
                G.add_edge(w1, w2, weight=data['weight'])
        
        return G

    def plot_word_network(G):
        plt.figure(figsize=(12, 8))

        # Tested multiple different layouts, these were the best configurations, each with their own merit
        # pos = nx.planar_layout(G, scale=1.3)
        # pos = nx.arf_layout(G, a=1.5, max_iter = 100)
        pos = nx.spring_layout(G, k=2.5, iterations=70, scale=1)

        thickness_weight = 1 / 15  # Multiplying by a scalar adjust the edge thickness
        edge_weights = [G[u][v]["weight"] * thickness_weight for u, v in G.edges()]

        
        clusters = list(nx.connected_components(G))
        colors = cm.rainbow(np.linspace(0, 1, len(clusters))) 

        # Coloring clusters to make things a bit easier to differentiate
        node_color_map = {}
        for color, cluster in zip(colors, clusters):
            for node in cluster:
                node_color_map[node] = color

        node_colors = [node_color_map[node] for node in G.nodes()]

        # Plot
        nx.draw(G, pos,
                node_color=node_colors,
                node_size=1000,
                font_size=8,
                font_weight='bold',
                width=edge_weights,
                with_labels=True,
                edge_color='gray',
                alpha=0.7)

        plt.title("Significant bigram connections as a network graph")
        return plt.gcf()

    all_tokens = list(chain.from_iterable(steam_reviews['even_cleaner_tokenized_review']))
    n_grams = list(ngrams(all_tokens, 2))
    freq_dist_ngrams = FreqDist(n_grams)

    G = create_network_visualization(freq_dist_ngrams, min_freq=2, max_edges=50)
    fig = plot_word_network(G)
    return fig
st.pyplot(plot_bigram_word_network())

st.write(
    """
    ... we can visualize these connections in a more intuitive fashion.

    We can notice that our most significant bigrams generally fall into a single cluster around the word "game".
    - Though we can try to gleam information from these clusters, because of how general this cluster's words are, there are no clear decisive insights to gain.

    From here, we categorize and filter.
    """
)

# These category wide keywords were created with the help of ChatGPT-4o on 12/3/24
# After creation of general keywords, we manually edited the listings for consistency, as well as added a few more game specific labels like "fps", "ost", "ui", "toxic" etc.
# Using https://gametree.me/gaming-terms/ as a reference
categories = {
    'Gameplay': [
        'gameplay', 'mechanics', 'controls', 'difficulty', 'balance', 'replayability',
        'progression', 'challenges', 'objectives', 'customization', 'levelling',
        'features', 'interaction', 'immersion', 'skills', 'pacing', 'grinding',
        'buff', 'nerf', 'op', 'meta', 'loot', 'skilltree', 'easteregg', 'camping',
        'dps', 'rpg', 'mmorpg', 'fps', 'ez'
    ],
    'Visuals': [
        'graphics', 'visuals', 'art', 'design', 'style', 'animation', 'environment',
        'lighting', 'textures', 'resolution', 'shaders', 'models', 'aesthetics',
        'realism', 'colors', 'visualfidelity', 'hud', 'ui', 'cutscenes'
    ],
    'Audio': [
        'music', 'soundtrack', 'audio', 'sound', 'effects', 'voice', 'ambiance',
        'backgroundmusic', 'sounddesign', 'narration', 'thememusic', 'voiceacting',
        'volumebalance', 'ost'
    ],
    'Performance': [
        'performance', 'optimization', 'framerate', 'bugs', 'glitches', 'crashes',
        'lag', 'latency', 'loadingtimes', 'stuttering', 'fps', 'systemrequirements',
        'compatibility', 'updates', 'patches', 'ping', 'framedrops', 'inputlag'
    ],
    'Narrative': [
        'story', 'plot', 'characters', 'lore', 'writing', 'quests',
        'storyline', 'campaign', 'worldbuilding', 'backstory',
        'characterdevelopment', 'narrativepacing', 'endings', 'twists', 'npc', 'sidequests'
    ],
    'Community': [
        'multiplayer', 'online', 'coop', 'community', 'friends', 'matchmaking',
        'social', 'guilds', 'chat', 'servers', 'forums',
        'playerbase', 'leaderboards', 'clans', 'competition', 'pvp', 'toxic',
        'esports', 'teamwork'
    ],
    'Accessibility': [
        'accessibility', 'interface', 'subtitles', 'tutorials', 'difficultymodes',
        'settings', 'colorblindmode', 'adjustability', 'menus',
        'shortcuts', 'tooltips', 'guides', 'userfriendly', 'onboarding', 'remapping',
        'assistmode'
    ],
    'Value': [
        'price', 'value', 'content', 'dlc', 'expansions',
        'microtransactions', 'rewards', 'unlockables', 'length',
        'longevity', 'variety', 'bundles', 'extras', 'paytowin',
        'freemium', 'grind'
    ]
}

def plot_common_words_by_category() -> plt.Figure:
    lemmatizer = WordNetLemmatizer()
    assigned_keywords = {} # Creating a dictionary of assigned keywords per category
    for category, keywords in categories.items():
        unique_keywords = []
        for word in keywords:
            if word not in assigned_keywords:
                assigned_keywords[word] = category
                unique_keywords.append(word)
        categories[category] = unique_keywords

    # To remain consistent with the lemmatized tokens of our cleaned reviews, we need to also lemmatize our category words
    lemmatized_categories = {
        category: [lemmatizer.lemmatize(word.lower()) for word in words]
        for category, words in categories.items()
    }

    # Reverse mapping from keyword to category
    keyword_to_category = {}
    for category, keywords in lemmatized_categories.items():
        for keyword in keywords:
            keyword_to_category[keyword] = category

    # Initialize counters for each category
    category_keyword_counts = {category: Counter() for category in categories}

    all_tokens = list(chain.from_iterable(steam_reviews['even_cleaner_tokenized_review']))

    # Count frequencies of keywords, ensuring no overlap
    for word in all_tokens:
        lemma_word = lemmatizer.lemmatize(word.lower())
        category = keyword_to_category.get(lemma_word)
        if category:
            category_keyword_counts[category][lemma_word] += 1

    top_keywords_per_category = {}
    for category, counter in category_keyword_counts.items():
        top_keywords = counter.most_common(10)  # Adjust the number as needed
        top_keywords_per_category[category] = top_keywords

    data = []
    for category, keywords in top_keywords_per_category.items():
        for word, count in keywords:
            data.append({'Category': category, 'Keyword': word, 'Frequency': count})

    df = pd.DataFrame(data)

    # Plot
    categories_list = list(categories.keys())
    num_categories = len(categories_list)
    fig, axes = plt.subplots(nrows=(num_categories + 1) // 2, ncols=2, figsize=(15, 20))
    axes: list[plt.Axes] = axes.ravel()

    for i, category in enumerate(categories_list):
        ax = axes[i]
        category_data = df[df['Category'] == category]
        sns.barplot(
            data=category_data,
            x='Frequency', y='Keyword',
            ax=ax,
            hue='Keyword',
            legend=False,
        )
        ax.set_title(category)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Keyword')
    plt.tight_layout()
    return fig
st.pyplot(plot_common_words_by_category())

st.write(
    """
    Here we plot the most common words by "category". We defined lists of known common words for different various common game development topics. See our curated list here.
    """
)
st.write("Categories:", categories)

st.write(
    """
    The frequency of the words have large bearing on their relative importance.
    - Interesting to note, under the performance category, 🌟users discuss 'bugs' much more often than any other performance issues!🌟
    """
)

def plot_common_words_by_category_stacked() -> plt.Figure:
    lemmatizer = WordNetLemmatizer()

    # Reverse mapping from keyword to category
    keyword_to_category = {
        keyword: category
        for category, keywords in categories.items()
        for keyword in keywords
    }

    # Initialize counters
    counters = {
        True: {category: Counter() for category in categories},
        False: {category: Counter() for category in categories}
    }

    # Count keyword occurrences for each recommendation group
    for _, row in steam_reviews.iterrows():
        tokens = row['even_cleaner_tokenized_review']
        recommended = row['recommended']
        for token in tokens:
            lemma_word = lemmatizer.lemmatize(token.lower())
            category = keyword_to_category.get(lemma_word)
            if category:
                counters[recommended][category][lemma_word] += 1

    # Prepare data for plotting
    plot_data = []
    for recommended, category_data in counters.items():
        for category, keyword_counts in category_data.items():
            for keyword, count in keyword_counts.items():
                plot_data.append({
                    'Recommended': recommended,
                    'Category': category,
                    'Keyword': keyword,
                    'Frequency': count
                })

    plot_df = pd.DataFrame(plot_data)

    # Aggregate frequencies by category
    category_summary = plot_df.groupby(['Category', 'Recommended'])['Frequency'].sum().unstack()

    # Plot stacked horizontal bar plot
    category_summary.plot(kind='barh', stacked=True, figsize=(12, 8))
    plt.title('Keyword Frequency by Category and Recommendation')
    plt.xlabel('Frequency')
    plt.ylabel('Category')
    plt.legend(title='Recommended', labels=['Not Recommended', 'Recommended'])
    plt.tight_layout()
    return plt.gcf()
st.pyplot(plot_common_words_by_category_stacked())

st.write(
    """
    To qualify the previous graph, we plot the overall frequency by category colored by whether the words are toward or against recommendation.

    We can notice an imbalance. Evidently, most of our keywords in the previous list are used in reviews which do not recommend the game.
    - Users are more likely to be specific in negative reviews, and be more general in positive reviews!
    """
)


def plot_word_clouds():
    # Define a set of words to exclude
    excluded_words = {"game", "good", "fucking", "shit", "play", "bad", "time", "fun", "great", "love", "amazing", "thing", "people", "fuck"}  # Add more words to exclude as needed

    # Filtering out words
    def filter_tokens(tokens, excluded_words):
        return [word for word in tokens if word.lower() not in excluded_words]

    # Filter tokens for both groups
    steam_reviews['filtered_tokens'] = steam_reviews['even_cleaner_tokenized_review'].apply(lambda tokens: filter_tokens(tokens, excluded_words))

    # Combine tokens into strings for each group
    true_text = ' '.join([' '.join(tokens) for tokens in steam_reviews[steam_reviews['recommended'] == True]['filtered_tokens']])
    false_text = ' '.join([' '.join(tokens) for tokens in steam_reviews[steam_reviews['recommended'] == False]['filtered_tokens']])

    # Create word clouds
    true_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(true_text)
    false_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(false_text)

    # Plot the word clouds
    plt.figure(figsize=(16, 8))

    # Word cloud for 'recommended == True'
    plt.subplot(1, 2, 1)
    plt.imshow(true_wordcloud, interpolation='bilinear')
    plt.title("WordCloud: Recommended = True", fontsize=16)
    plt.axis('off')

    # Word cloud for 'recommended == False'
    plt.subplot(1, 2, 2)
    plt.imshow(false_wordcloud, interpolation='bilinear')
    plt.title("WordCloud: Recommended = False", fontsize=16)
    plt.axis('off')

    plt.tight_layout()
    return plt.gcf()
st.pyplot(plot_word_clouds())

st.write(
    """
    We can visualize common words using an alternative visualization method: the word cloud. We notice similar commonalities with our previous bar graphs for common words.

    We may want to filter our word cloud to exclude the more uninteresting words. We would like to observe the frequency of words with direct relevance to aspects of the games they describe.
    """
)

def plot_author_playtime_by_recommendation_distributions():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes: list[plt.Axes] = axes
    sns.boxplot(
        data=steam_reviews,
        x="recommended",
        y="author_playtime_at_review",
        hue="recommended",
        legend=False,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_title('Author playtime at review distribution by recommendation')
    axes[0].set_xlabel('Recommended')
    axes[0].set_ylabel('Author playtime at review (minutes)')

    sns.kdeplot(data=steam_reviews[steam_reviews['recommended'] == True], 
                x='author_playtime_at_review', label='Recommended', log_scale=True, ax=axes[1])
    sns.kdeplot(data=steam_reviews[steam_reviews['recommended'] == False], 
                x='author_playtime_at_review', label='Not Recommended', log_scale=True, ax=axes[1])
    axes[1].set_title('Density plot of author playtime by recommendation')
    axes[1].set_xlabel('Author playtime at review (minutes) (log scaled)')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    return fig
st.pyplot(plot_author_playtime_by_recommendation_distributions())

st.write(
    """
    To answer whether playtime has bearing on our recommendation, we plot two graphs: A double boxplot and a double kernel density estimate line.
    - We use log scaling to account for our heavy skew to the right. We do indeed find better results in this scaling, we should however consider this is more exponential.

    From this, we can notice that authors that recommend a game typically have a higher amount of playtime than authors that do not.
    - Intuitively, this aligns with the notion of, players who enjoy a game would typically play more, and players who do not enjoy a game may quit out earlier. They may not want to continue.
    - From this we can draw insight: players with a lot to say typically enjoyed the game!
    """
)

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
            **🌟Informative Keyword Frequencies🌟**
            """
        )

        @st.fragment()
        def dashboard_plot_word_clouds():
            st.write("Type any additional exclusions as comma separated words!")

            more_exclusions_text = st.text_input("Additional Exclusion List:", help="EX: player, fun, great")

            excluded_words = [
                "game",
                "good",
                "fucking",
                "shit",
                "play",
                "bad",
                "time",
                "love",
                "amazing",
                "thing",
                "people",
                "fuck",
            ]  # Add more words to exclude as needed, trying to exclude things like swear words from production
            more_exclusions = [item.strip() for item in more_exclusions_text.split(",")]

            st.write("Current filter:", more_exclusions)

            excluded_words = excluded_words + more_exclusions
            def filter_tokens(tokens, excluded_words):
                return [word for word in tokens if word.lower() not in excluded_words]

            steam_reviews['filtered_tokens'] = steam_reviews['even_cleaner_tokenized_review'].apply(lambda tokens: filter_tokens(tokens, excluded_words))

            # Combining for word cloud usage
            true_text = ' '.join([' '.join(tokens) for tokens in steam_reviews[steam_reviews['recommended'] == True]['filtered_tokens']])
            false_text = ' '.join([' '.join(tokens) for tokens in steam_reviews[steam_reviews['recommended'] == False]['filtered_tokens']])

            true_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(true_text)
            false_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(false_text)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            axes[0].imshow(true_wordcloud, interpolation='bilinear')
            axes[0].set_title("WordCloud: Recommended", fontsize=16)
            axes[0].axis('off')
            
            
            axes[1].imshow(false_wordcloud, interpolation='bilinear')
            axes[1].set_title("WordCloud: Not Recommended", fontsize=16)
            axes[1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

        dashboard_plot_word_clouds()

        st.write(
            """
            Keywords for positive and negative reviews:
            - Look out for story and narrative aspects, these keywords are frequent in both good and bad reviews!
            - Players seem to especially hate bugs and crashes!
            - Play around with the filter to try and gleam other common user talking points by recommendation.
            """
        )