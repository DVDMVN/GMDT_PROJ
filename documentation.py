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
    - `app_id`: The ID of the product as allocated by Steam
    - `name`: Game title
    - `short_description`: A small blurb that appears on the product page under the header image
    - `developer`: Game developer(s)
    - `publisher`: Game publisher(s)
    - `genre`: Genre(s) of the game ('Action', 'Adventure', 'Strategy', etc.)
    - `tags`: User assigned tags for the game ('Action', 'Multiplayer', 'Shooter', etc.)
    - `type`: Whether the product is a game or some sort of sortware that is not a game
    - `categories`: Categories / features the product has ('Single-player', 'PvP', 'Online', etc.)
    - `owners`: An approximate$^1$ estimate of the number of users who own the game according to Steam Spy
    - `positive_ratings`: Number of positive ratings
    - `negative_ratings`: Number of negative ratings
    - `price`: Price of the game in USD
    - `initial_price`: Price of the game in USD at launch
    - `discount`: Percentage of sale at the time of the snapshot
    - `ccu`: Peak concurrent player count for the game
    - `languages`: Languages supported by the game
    - `release_date`: Release date of the game
    - `required_age`: The required user age minimum for the game
    - `website`: The website of the developer/publisher of the product
    - `header_image`: A link to the header image of the game
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