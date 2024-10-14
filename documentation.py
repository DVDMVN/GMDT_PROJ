import streamlit as st
st.header("Appendix")

# Attribute descriptions for each dataset
st.subheader("Attribute List Descriptions for Each Dataset")

st.write("**Top 1500 Steam Games Dataset Attributes**")
st.markdown(
    """
    - `name`: Game title
    - `release_date`: Release date of the game
    - `copies_sold`: Total number of copies sold (estimated<sup>1</sup>)
    - `price`: Original retail price of the game at release
    - `revenue`: The amount of money generated from the sales of the game (estimated<sup>1</sup>)
    - `review_score`: The score or rating given to the game based on user and critic reviews (out of 100)
    - `publisher_class`: A classification of the publisher, indicating whether the publisher is AAA, AA or indie
    - `publisher`: Game publisher name(s)
    - `developer`: Game developer name(s) or studio name
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