import streamlit as st
st.header("Appendix")

st.divider()

st.write(
    """
        Most of our datasets deal with Steam API data, or derivative 3rd party API data. Our ground truths described in our attribute lists come from a combination
        of inference, Steam API return type documentation, and kaggle author dataset documentation.
        - The Steam API notorious for its poor documentation. Much of the API is undocumented through their official channels. For this reason, we also utilized 
        3rd party unofficial documentation for reference in data types and description.

        Original kaggle author documentation pages:
        - [Top 1500 games on steam by revenue 09-09-2024 (kaggle.com)](https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024)
        - [All 55000 Games on Steam November 2022](https://www.kaggle.com/datasets/tristan581/all-55000-games-on-steam-november-2022)
        - [Metacritic Reviews 1995 - 2024](https://www.kaggle.com/datasets/beridzeg45/video-games)
        - [Steam Reviews 2021](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021)
    
        Steam API documentation pages:
        - [Unofficial Documentation for InternalSteamWebAPI](https://github.com/Revadike/InternalSteamWebAPI)
        - [Unofficial Searchable Documentation for public SteamWebAPI](https://steamapi.xpaw.me/)
        - [Official Documentation for User Reviews](https://partner.steamgames.com/doc/store/getreviews)
    """
)

st.divider()

# Attribute descriptions for each dataset
st.subheader("Attribute List Descriptions for Each Dataset")

st.write("**Top 1500 Steam Games Dataset Attributes**")
st.write(
    """
    - `name`: Game title
    - `release_date`: Release date of the game
    - `copies_sold`: Total number of copies sold (estimated [1])
    - `price`: Original retail price of the game at release
    - `revenue`: The amount of money generated from the sales of the game (estimated [1])
    - `review_score`: The score or rating given to the game based on user and critic reviews (out of 100)
    - `publisher_class`: A classification of the publisher, indicating whether the publisher is AAA, AA or indie
    - `publisher`: Game publisher name(s)
    - `developer`: Game developer name(s) or studio name
    """
)

st.write("**All 55,000 Steam Games Dataset Attributes**")
st.write(
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
st.write(
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

st.write("**Steam Reviews Dataset Attributes**")
st.write(
    """
    - `app_id`: The ID of the product as allocated by Steam
    - `name`: Game title
    - `review_id`: The ID of the review as allocated by Steam
    - `language`: The language in which the review was written
    - `review`: Text content of the review
    - `timestamp_created`: The date and time when the review was originally posted
    - `timestamp_updated`: The date and time when the review was last edited
    - `recommended`: Boolean indicating if the user recommends the game (True) or not (False)
    - `votes_helpful`: Number of times other users marked the review as helpful
    - `votes_funny`: Number of times other users marked the review as funny
    - `weighted_vote_score`: A score reflecting the overall helpfulness of the review, adjusted by Steam in a hidden algorithm
    - `comment_count`: Number of comments made on the review by other users
    - `steam_purchase`: Boolean indicating whether the user purchased the game on Steam
    - `received_for_free`: Boolean indicating whether the user received the game for free
    - `written_during_early_access`: Boolean indicating whether the review was written while the game was in early access
    - `author_steamid`: The ID of the author of the review as allocated by Steam
    - `author_num_games_owned`: Total number of games owned by the review author
    - `author_num_reviews`: Total number of reviews written by the author
    - `author_playtime_forever`: Total playtime (in minutes) the author has logged on the game
    - `author_playtime_last_two_weeks`: Playtime (in minutes) the author has logged in the last two weeks
    - `author_playtime_at_review`: Playtime (in minutes) the author had at the time of writing the review
    - `author_last_played`: The timestamp of when the author last played the game
    - `review_text_length`: The length of the review text, measured in characters or words
    """
)

st.divider()

# Some additional sources to provide integrity to some ideas
st.subheader("Additional Sources")

st.write(
    """
    [1] [Meet ConcernedApe: The Master Behind Stardew Valley and Beyond - thinglabs](https://thinglabs.io/meet-concernedape-the-master-behind-stardew-valley-and-beyond)

    [2] [2012 Video Game Awards Winners (collider.com)](https://collider.com/2012-video-game-awards-winners-announced-the-walking-dead-named-game-of-the-year-halo-4-wins-best-xbox-game/)

    [3] [Why Are RPGs Popular? A Stunning World Awaits | by TJ | Medium](https://nitemare121.medium.com/why-are-rpgs-popular-a-stunning-world-awaits-3f629adfa1b)

    [4] [Infographic: Indie game revenues on Steam | Video Game Insights](https://vginsights.com/insights/article/infographic-indie-game-revenues-on-steam)

    [5] [It's time for MetaCritic to stand up to review-bombings | GamesRadar+](https://www.gamesradar.com/its-time-for-metacritic-to-stand-up-to-review-bombings/)

    [6] [The Evolution of Gaming: From Traditional Consoles to the Rise of Online Platforms - PlayStation Universe](https://www.psu.com/news/the-evolution-of-gaming-from-traditional-consoles-to-the-rise-of-online-platforms/)

    [7] [The Rise of Indie Games: How Small Studios Are Making Big Waves | by TechGamer Nexus | Medium](https://medium.com/@techgamernexus/the-rise-of-indie-games-how-small-studios-are-making-big-waves-46f6c495bf42#:~:text=The%20indie%20game%20revolution%20began%20in%20earnest%20in,traditional%20gatekeepers%20and%20reach%20a%20global%20audience%20directly.)

    [8] [GDC Vault - The Making of 'Divinity: Original Sin 2'](https://gdcvault.com/play/1026452/The-Making-of-Divinity-Original)

    [9] [The Best Metroidvania Games Ever | Den of Geek](https://www.denofgeek.com/games/the-best-metroidvania-games-ever/)

    [10] [Best Metroidvania games to explore to absolute completion | GamesRadar+](https://www.gamesradar.com/best-metroidvania-games/)

    [11] [EA’s Staggered Release Experiment | Video Game Law](https://videogamelaw.allard.ubc.ca/2019/09/11/eas-staggered-release-experiment/)

    [12] [Strategically Timing Your Game Release – Sibs Video Game Consulting](https://sibs.llc/strategically-timing-your-game-release/)

    [13] [The Curse of Knowledge Phenomenon](https://en.wikipedia.org/wiki/Curse_of_knowledge)

    [14] [Lifting the Curse of Knowledge](https://daedtech.com/lifting-the-curse-of-knowledge/)

    [15] [How Can You Use Lemmatization to Improve Your Text Data?](https://www.linkedin.com/advice/1/how-can-you-use-lemmatization-improve-your-text-data-wmpxe)
    
    [16] [Feature Importance in XGBoost Models](https://mljourney.com/xgboost-feature-importance-comprehensive-guide/)

    [17] [Is Story Important In Video Games?](https://www.gameskinny.com/culture/is-story-important-in-video-games-it-is-not-that-simple/)

    [18] [Developer Experience and Video Game Success](https://arxiv.org/abs/1801.04293)
    """
)

st.divider()

st.write("This site is open source, to view the source code please visit [GitHub](https://github.com/DVDMVN/GMDT_PROJ).")