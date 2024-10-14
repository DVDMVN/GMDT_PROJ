import streamlit as st
from utils import load_datasets
data = load_datasets()
top_1500_steam, all_55000_steam, metacritic = data['top_1500_steam'], data['all_55000_steam'], data['metacritic']

st.title("About the Data")

st.header("Introduction")

st.write("")

st.header("🌟Goal🌟")

st.write(
    """
        The goal of my project is to identify key factors of game success using data driven tools and insights to enable current game developers to make more informed decisions for their development strategy.
    """
)

# Datasets Overview
st.header("Datasets Overview")

st.write(
    """
    In this project, we use three distinct datasets that provide complementary perspectives on game success:
    - [Top 1500 games on steam by revenue 09-09-2024 (kaggle.com)](https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024)
    - [All 55000 Games on Steam November 2022](https://www.kaggle.com/datasets/tristan581/all-55000-games-on-steam-november-2022)
    - [Metacritic Reviews 1995 - 2024](https://www.kaggle.com/datasets/beridzeg45/video-games)
    """
)

# Displaying each dataset with basic information
st.subheader("Dataset Previews")

# Display the first few rows of each dataframe
st.write("##### Top 1500 Steam Games Dataset")
st.dataframe(top_1500_steam.head())

st.write("##### All 55,000 Steam Games Dataset")
st.dataframe(all_55000_steam.head())

st.write("##### Metacritic Reviews Dataset")
st.dataframe(metacritic.head())

# Display basic statistics and shapes
st.subheader("Basic Statistics and Shapes of the Datasets")

def show_basic_statistics(df, title):
    st.write(f'### {title}')
    col1, col2 = st.columns(2)
    with col1:
        st.write(df.describe())

    with col2:
        object_cols = df.select_dtypes(include=['object']).columns
        st.write(df[object_cols].describe())
    st.write(f"Shape: {df.shape}")

show_basic_statistics(top_1500_steam, "Top 1500 Steam Games Dataset")
show_basic_statistics(all_55000_steam, "All 55,000 Steam Games Dataset")
show_basic_statistics(metacritic, "Metacritic Reviews Dataset")

# STANDARDIZATION
st.header("Data Cleaning and Preparation")

st.write(
    """
    In order to make our datasets consistent and ready for analysis, the following data cleaning steps were performed:
    - **Standardized Naming**: All feature names were standardized to lowercase with underscores separating words.
    - **Date Standardization**: The release date format was standardized to "YYYY-MM-DD".
    """
)

# MISSING DATA ANALYSIS
st.subheader("Missing Data Analysis")

st.write(
    """
    Missing data was present across all datasets. We approached this issue by:
    - Dropping rows with missing values for essential features like `name` and `release_date`.
    - Imputing missing values in numerical columns using median imputation.
    - Keeping rows with missing values in certain categorical features to avoid losing potentially valuable information.
    """
)

# JOINING STRATEGY
st.header("Data Merging Strategy")

st.write(
    """
    - We first standardized game titles across all datasets to ensure accurate merging.
    - We performed a left join using `name` as the key to merge the `top_1500_steam` and `metacritic` datasets, retaining all observations in `top_1500_steam`.
    - We further joined the resulting dataframe with `all_55000_steam` using a similar left join strategy, incorporating all attributes relevant to our analysis.
    """
)

# CONCLUSIONS
st.header("Conclusions")

st.write(
    """
    After merging and cleaning the datasets, the final dataframe used in the analysis contains:
    - The `top_1500_steam` games dataset as the primary base.
    - Additional review information from the `metacritic` dataset.
    - Supplementary attributes from the `all_55000_steam` dataset.
    
    This comprehensive dataset allows for a holistic analysis of the factors influencing game success, with attributes ranging from user reviews and ratings to revenue and release information.
    """
)