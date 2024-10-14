import streamlit as st
st.header("Appendix")

# Attribute descriptions for each dataset
st.subheader("Attribute List Descriptions for Each Dataset")

st.write("**Top 1500 Steam Games Dataset Attributes**")
st.markdown(
    """
    - `name`: Game title
    - `release_date`: Release date of the game
    - `developer`: Game developer(s)
    - `publisher`: Game publisher(s)
    - `price`: Price of the game
    - `revenue`: An estimate of total revenue generated
    - `owners`: An estimate of the number of game owners
    - `genre`: Genre(s) of the game
    """
)

st.write("**All 55,000 Steam Games Dataset Attributes**")
st.markdown(
    """
    - `name`: Game title
    - `release_date`: Release date of the game
    - `developer`: Game developer(s)
    - `publisher`: Game publisher(s)
    - `positive_ratings`: Number of positive ratings
    - `negative_ratings`: Number of negative ratings
    - `average_playtime`: Average playtime in minutes
    - `price`: Price of the game
    - `owners`: Number of game owners
    - `genre`: Genre of the game
    """
)

st.write("**Metacritic Reviews Dataset Attributes**")
st.markdown(
    """
    - `title`: Game title
    - `release_date`: Release date of the game
    - `developer`: Game developer(s)
    - `publisher`: Game publisher(s)
    - `genres`: Genre of the game
    - `product_rating`: Game rating
    - `user_score`: User review score
    - `user_ratings_count`: Number of user ratings
    """
)