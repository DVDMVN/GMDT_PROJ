import streamlit as st

st.set_page_config(
    page_title="GMDT_PROJ",
    page_icon="🎮",
)

welcome = st.Page("welcome.py", title="Welcome", icon="👋")
about = st.Page("about.py", title="About the Data", icon="💾")
top_1500_steam_analysis = st.Page("top_1500_steam_analysis.py", title="Top Steam Games Analysis", icon="🏆")
all_55000_steam_analysis = st.Page("all_55000_steam_analysis.py", title="All Steam Games Analysis", icon="🌏")
metacritic_analysis = st.Page("metacritic_analysis.py", title="Metacritic Reviews Analysis", icon="🕹️")
steam_reviews_analysis = st.Page("steam_reviews_analysis.py", title="Steam Reviews Analysis", icon="💬")
documentation = st.Page("documentation.py", title="Documentation", icon="📔")

pg = st.navigation([welcome, about, top_1500_steam_analysis, all_55000_steam_analysis, metacritic_analysis, steam_reviews_analysis, documentation])
pg.run()