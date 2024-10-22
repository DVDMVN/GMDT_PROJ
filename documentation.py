import streamlit as st
st.header("Appendix")

st.divider()

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
    """
)

st.divider()

st.write("This site is open source, to view the source code please visit [GitHub](https://github.com/DVDMVN/GMDT_PROJ).")