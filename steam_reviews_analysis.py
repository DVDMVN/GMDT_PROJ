import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from plotly.subplots import make_subplots

from utils import load_steam_reviews
steam_reviews = load_steam_reviews()
plt.style.use('ggplot')

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
2. [Feature Explorations](#feature-explorations)
    * The exploration journey used to uncover key insights. Jumping to our dashboard is an option, but we _encourage_ a review of our exploration to really grasp understanding
    of our findings.
    * This section introduces some basic statistics before branching into tabs for particular feature dives, allowing you to choose specific branches of exploration based on your interests.
3. [Key Insights](#key-insights)
    * A comprehensive list of our most important findings, along with some caveats to consider.
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
        - What does our text data look like?
        - What are the most common positive reviews?
        - What are the most common complaints in reviews that have a negative recommendation?
        - Does playtime have any bearing on our recommendation? Does playtime have bearing on our review text shape?
            - Can we consider extremely short or extremely long playtime to be a good indicator of an outlier review?
    """
)
