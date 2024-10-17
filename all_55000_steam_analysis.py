import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_all_55000_steam

all_55000_steam = load_all_55000_steam()

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

st.write(
    """
        Chart of publisher class distribution
    """
)