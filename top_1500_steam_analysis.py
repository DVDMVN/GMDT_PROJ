import streamlit as st

st.header("Exploring the top Steam games")

st.write(
    """
    In this page, we will explore the top 1500 steam games dataset, a dataset containing the 1500 most profitable games on steam from this year (2024), up to Sept. 9th.
    The data was collected via the site "gamalytic.com".
    
    Before getting into the analysis, let's first explain our key features and how we intend to utilize them. We must also explain some various caveots regarding the 
    integrity of some particular features.
    """
)

st.write(
    """
    We utilize all features of this dataset in at least some way. Here we list specific highlighted features and a short description of their usage:
    - `release_date`: Release date of the game
        - Using release_date, we can ask: how does the release_date affect revenue?
    - `copies_sold`: Total estimated$^1$ number of copies sold
    - `price`: Original retail price of the game at release$^2$
    - `revenue`: The estimated$^1$ amount of money generated from the sales of the game
    - `review_score`: The score or rating given to the game based on user and critic reviews (out of 100)
    - `publisher_class`: A classification of the publisher, indicating whether the publisher is AAA, AA, indie, or hobbyist
    - `publisher`: Game publisher name(s)
    - `developer`: Game developer name(s) or studio name

    See our documentation page for further details on this dataset's features set:
    """
)

st.page_link("pages/documentation.py", label="Documentation", icon="")

st.write(
    """
    Some important caveots:
    1. The copies_sold and revenue features from Gamalytic are only estimates, the exact figures are kept private. 
    Gamalytic uses a complex algorithm for calculating both of these figures, see more here: (Link)[https://gamalytic.com/blog/how-to-accurately-estimate-steam-sales]
    """
)

st.subheader("Qualitative Trend Analysis (Visualization)")

st.write(
    """
    We begin with some data breakouts.
    """
)



