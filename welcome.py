import streamlit as st

st.title("Welcome to the Game Market Insights Dashboard")
st.header("Discover Data-Driven Trends in Game Development")

st.subheader("About this Project")
st.write("""
    This application is designed to provide valuable insights into the gaming market by analyzing multiple datasets related to game sales, user ratings, and marketplace trends.
    Our goal is to assist game developers in making data-informed decisions by showcasing key factors that influence game success.
""")

st.write("""
    Contents:
    - This dashboard utilizes 3 distinct game market datasets from kaggle. Each dataset having unique attributes, we separated 
    analysis for each between different pages.
    - Learn more about our datasets and their attributes in the documentation and about pages.
""")

st.markdown(
    """
    ---
    Ready to dive into the data? Choose an option from the sidebar to start exploring!  
    """
)

#TODO: Insert some animated graphs as a teaser here

st.write("For more information or to view the full documentation, please visit [GitHub](https://github.com/DVDMVN/GMDT_PROJ).")