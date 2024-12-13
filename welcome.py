import streamlit as st

st.title("Welcome to the Game Market Insights Dashboard")
st.header("Discover Data-Driven Trends in Game Development", divider=True)

st.subheader("About this Project")
st.write("""
    This application is designed to provide valuable insights into the gaming market by analyzing multiple datasets related to game sales, user ratings, and marketplace trends.
    Our goal is to assist game developers in making data-informed decisions by showcasing key factors that influence game success.
""")

st.write("""
    Contents:
    - This dashboard utilizes 4 distinct game market datasets from kaggle. Each dataset having unique attributes, we separated 
    analysis for each between different pages.
    - Learn more about our datasets, their attributes, and handling of data quality issues in the documentation and about pages.
    - Explore our data visualizations and insights in our individual analysis pages.
""")

st.write("""
    Our analysis followed a common systematic approach, the following figure represents a rough order of our steps:
""")

st.image("img/schema.png")

st.write("""
    - We discuss loading, preprocessing and general IDA for all datasets in our about page.
    - Feature engineering, EDA and modeling are addressed on separate pages for each dataset.
""")

# st.write("""
#     - Loading, preprocessing, and IDA for all datasets is addressed in the "About the Data" page.
#         - Preprocessing encapsulates handling of lots of different data quality issues, such as date formatting and other value standardizations, 
#          missingness analysis, handling of missing values, outlier analysis, and handling of outliers.
#     - Feature engineering, EDA and modeling are addressed on separate pages for each dataset.
# """)

st.markdown(
    """
    ---
    Ready to dive into the data? Choose an option from the sidebar to start exploring!  
    """
)

#TODO: Insert some animated graphs as a teaser here

st.write("For more information or to view the full documentation, please visit [GitHub](https://github.com/DVDMVN/GMDT_PROJ).")