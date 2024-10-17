import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import load_datasets
data = load_datasets()
top_1500_steam, all_55000_steam, metacritic = data['top_1500_steam'], data['all_55000_steam'], data['metacritic']

st.title("About the Data")

st.write(
    """
    This page details the process for cleaning and preprocessing of the data before its usage in analysis. There are many choices made with this dataset in regards to standardization,
    encoding, and imputing. Though each choice is made with ample reasoning and investigation, noting why changes were made may be useful in the assessment of our analysis.
    """
)

# Datasets Overview
st.header("Datasets Overview", divider=True)

st.write(
    """
    In this project, we use three distinct datasets that provide complementary perspectives on game success:
    - [Top 1500 games on steam by revenue 09-09-2024 (kaggle.com)](https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024)
    - [All 55000 Games on Steam November 2022](https://www.kaggle.com/datasets/tristan581/all-55000-games-on-steam-november-2022)
    - [Metacritic Reviews 1995 - 2024](https://www.kaggle.com/datasets/beridzeg45/video-games)
    """
)

# Displaying each dataset with basic information
st.subheader("Dataset Previews", divider=True)

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
    tab_1, tab_2 = st.tabs(["Numerical Statistics", "Categorical Statistics"])
    with tab_1:
        st.write(df.describe())

    with tab_2:
        object_cols = df.select_dtypes(include=['object']).columns
        st.write(df[object_cols].describe())
    st.write(f"##### Shape: {df.shape}")

show_basic_statistics(top_1500_steam, "Top 1500 Steam Games Dataset")
show_basic_statistics(all_55000_steam, "All 55,000 Steam Games Dataset")
show_basic_statistics(metacritic, "Metacritic Reviews Dataset")

# STANDARDIZATION
st.header("Data Cleaning and Preprocessing")

st.subheader("Data Standardization", divider=True)

st.write(
    """
    In order to make our datasets consistent and ready for analysis, the following data cleaning steps were performed:
    - **Feature Names**: All feature names were standardized to lowercase with underscores separating words.
    - **Date Format**: Release dates were standardized to "YYYY-MM-DD" format.
    - **Review Score**: Review scores were standardized to use a 100 point scale (0 - 100)
    - **Pricing**: Prices were standardized to utilize floating point format (999 -> 9.99)
    - **List Data**: Many of the features within our datasets have value which are string literals of lists and dictionaries. We reformatted these to their original form
    where neccessary
    """
)

# MISSING DATA ANALYSIS
st.subheader("Missing Data Analysis")

pre_top_1500_steam = pd.read_csv("data/pre_top_1500_steam.csv")
pre_all_55000_steam = pd.read_csv("data/pre_all_55000_steam.csv")
pre_metacritic = pd.read_csv("data/pre_metacritic.csv")

st.write(
    """
    Previews of our missingness:
    """
)

pre_top_1500_steam_nulls = pd.DataFrame({'Null Counts': pre_top_1500_steam.isnull().sum()})
pre_all_55000_steam_nulls = pd.DataFrame({'Null Counts': pre_all_55000_steam.isnull().sum()})
pre_metacritic_nulls = pd.DataFrame({'Null Counts': pre_metacritic.isnull().sum()})

top_1500_steam_nulls = pd.DataFrame({'Null Counts': top_1500_steam.isnull().sum()})
all_55000_steam_nulls = pd.DataFrame({'Null Counts': all_55000_steam.isnull().sum()})
metacritic_nulls = pd.DataFrame({'Null Counts': metacritic.isnull().sum()})

tab1, tab2 = st.tabs(['Unprocessed', 'Postprocessing'])
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Top 1500 Steam Games Missing Values Count")
        st.dataframe(pre_top_1500_steam_nulls)
    with col2:
        st.write("All 55000 Steam Games Missing Values Count")
        st.dataframe(pre_all_55000_steam_nulls)
    with col3:
        st.write("Metacritic Reviews Missing Values Count")
        st.dataframe(pre_metacritic_nulls)
with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Top 1500 Steam Games Missing Values Count")
        st.dataframe(top_1500_steam_nulls)
    with col2:
        st.write("All 55000 Steam Games Missing Values Count")
        st.dataframe(all_55000_steam_nulls)
    with col3:
        st.write("Metacritic Reviews Missing Values Count")
        st.dataframe(metacritic_nulls)

st.write(
    """
    Our datasets exhibit lots of missing values of varying amounts and for various different features:
    - Within top_1500_steam we have very few missing values, only 1 or 2 for publishers and developers.
    - Within all_55000_steam we have missing values for many different features: short_description, developer, publisher, genre, tags, categories, languages, release_date, and website
    - Within metacritic we have a large proportion of missing values, as well as missing values for almost every feature across the board.

    To determine how best to handle our missing data, we must performed extensive analysis. Within this
    next section, we showcase our exploration into the missingness, and handling of our missing data. Though you may skip this section and simply
    read our conclusions, we highly recommend reviewing this information to gain a better understanding of our decisions.
    """
)

top_missing_tab, all_missing_tab, meta_missing_tab = st.tabs(["Top 1500 Steam Handling", "All 55000 Steam Handling", "Metacritic Handling"])

with top_missing_tab:
    st.write("With the top 1500 Steam games dataset, since we have only a few missing values we can simply examine every row:")

    st.dataframe(pre_top_1500_steam[pre_top_1500_steam.isnull().any(axis=1)])

    st.write(
        """
        We can note that all our observations with missing values are "indie" or "hobbyist" type games. 
        These games are likely self-published, and so our publisher and developer are likely to be the same. 
        To remedy this, since there are so few rows, we will simply manually fill in the missing values using their steam page information.
        """
    )

    code = '''
        top_1500_steam.loc[[643, 765], 'developers'] = ['Lovely Games', 'Naku Kinoko']
        top_1500_steam.loc[710, 'publishers'] = 'SWDTech Games'
        top_1500_steam.loc[[643, 710, 765]]
    '''
    st.code(code, language='python')

    st.dataframe(top_1500_steam.loc[[643, 710, 765]])

with all_missing_tab:
    st.write(
        """
        For the all_55000_steam dataset, we have a varying amount of missing data between our features, some have large proportions while others have very few. From our previous basic null count, 
        we can see our 'website' feature has a disproportionately high missing value count. Since our analysis will not be particular on the site value itself, we will not be attempting to 
        impute any values for this feature.
        
        In addition to website, we are also not utilizing the categories column (though the name 'categories' seems to indicate some importance, the categories for which this feature is referring to are more akin to
        'technologies provided', which is not part of our analysis as of yet). Our goal in handling is to try and save our developers, publishers and release_dates as much as possible.
        """
    )
    def plot_missingness_for_all_steam():
        all_55000_steam_nulls_no_website = pre_all_55000_steam_nulls.drop(['website'], axis=0)
        fig = plt.figure(figsize=(10,5))
        plt.bar(
            x = all_55000_steam_nulls_no_website[all_55000_steam_nulls_no_website['Null Counts'] != 0].index,
            height = all_55000_steam_nulls_no_website[all_55000_steam_nulls_no_website['Null Counts'] != 0]['Null Counts'],
        )
        plt.title("Null value frequency for feature in all 55000 steam dataset")
        return fig
    st.pyplot(plot_missingness_for_all_steam())

    st.write(
        f"""
        Number of observations with missing data (not website): {pre_all_55000_steam.drop(labels=['website'], axis=1)[pre_all_55000_steam.drop(labels=['website'], axis=1).isnull().any(axis=1)].__len__()}
        """
    )

    st.write(
        """
        Within our features with missing values, our frequency proportions are relatively low; we have a couple hundred or so missing values, with categories being an outlier at around a thousand. 
        To understand whether these missing values are by accident or if they are truly missing from the game's details, we performed some manual analysis for a few observations, comparing 
        details missing to their active game pages on Steam.
        """
    )
    st.dataframe(pre_all_55000_steam.drop(labels=['website'], axis=1)[pre_all_55000_steam.drop(labels=['website'], axis=1).isnull().any(axis=1)].tail())

    st.write(
        """
        The "SUBURI" observation is missing its categories feature, however from a quick glance at their Steam page, they seem to have categories for their game listed. Doing some further analysis reveals the same for many other observations within our dataset, features missing which are present.
        - This may be due to the age of the dataset, games may have updated their page since their release.

        Because the amount of missing data is relatively low, and our missing features can be found on the game's steam page, we may utilize the Steam API to restore as many features as possible.
        - Some features may still remain NaN, the data may truly be missing from the Steam game details.
        """
    )
    code = """
from utils import get_steam_app_details_from_app_id
from datetime import datetime

def clean_and_save():
    cleaned_app_details_dict = {}
    for app_id, app_details in app_details_dict.items():
        if app_details is not None:
            cleaned_app_details_dict[app_id] = app_details
    app_details_df = pd.DataFrame(index=cleaned_app_details_dict.keys(), data=cleaned_app_details_dict.values())
    app_details_df.to_csv('raw_data/loaded_steam_app_details.csv')
    return app_details_df


# Get information for all of our observations which have missing data (that is not website)
app_ids = all_55000_steam.drop(labels=['website'], axis=1)[all_55000_steam.drop(labels=['website'], axis=1).isnull().any(axis=1)]['app_id'].values

app_details_dict = {}
loaded_app_details = 0
for app_id in app_ids:
    app_details_dict[app_id] = get_steam_app_details_from_app_id(str(app_id))
    loaded_app_details += 1
    if loaded_app_details % 100 == 0: # Every 100, save a log
        with open('steam_api_logging.txt', 'a', encoding='utf-8') as f:
            f.write(f"Successfully loaded {loaded_app_details} app details | time: {datetime.now()}\n")
            f.close()
    if loaded_app_details % 200 == 0: # Every 200, save a snapshot
        clean_and_save()
clean_and_save()
    """
    st.code(code, language='python')

    st.write(
        """
        Within utils.py
        """
    )

    code = """
@retry(max_retries=10, delay=30, backoff=1.05, logfile='steam_api_logging.txt', exceptions=(requests.exceptions.RequestException, Exception))
def get_steam_app_details_from_app_id(app_id: str) -> dict:
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    response = requests.get(url)
    data = response.json()
    
    if data and data[app_id] and data[app_id]['success'] == True:
        app_details = data[app_id]['data']
        return app_details
    else:
        raise Exception(f"Failed to retrieve data for app ID {app_id}")    
    
"""
    st.code(code, language='python')

    st.write(
        """
        According to our logs, we encountered 4 truly missing app details.

        Since the amount was minimal, we opt for a manual review for each app_id utilizing steamdb.info

        - App ID 499450: "The Witcher 3: Wild Hunt - Game of the Year Edition"
            - This app was retired from the steam store some time ago.
        - App ID 813350: "Ben 10 VR"
            - This app is not available within our region of polling.
        - App ID 895970: "BlackShot Revolution"
            - This app is not available within our region of polling.
        - App ID 974470: "Girls Free"
            - This app was retired from the steam store some time ago.
        """
    )

    st.write(
        """
        We now utilize our pulled app details to impute values wherever possible.

        Reanalyzing, we find that we only gained a minor improvement in NaN count, we are still missing many rows. Evidently the steam API can give us missing values, which may explain the dataset's missingness origins.
        - Checking several observations manually, the values are truly missing from our Steam API call, despite other information being returned.

        We cannot fix by true imputation any further than this, but from our observations we may yet improve our result. Looking at our data, evidently we can see publishers and developers are often one and the same. In this light, we can reasonably impute publisher by developer (if present) and developer by publisher (if present).
        """
    )

    st.code("""
# Checking our proportion which has publishers developers as the same entry
publishers_developers_same = all_55000_steam['publishers'] == all_55000_steam['developers']
percentage_publishers_developers_same = publishers_developers_same.sum() / all_55000_steam.__len__()
print(f"Percentage of games with the same publisher and developer: {percentage_publishers_developers_same * 100:.2f}%")

# Checking where our publisher is missing but the developer is present
missing_publishers_present_developers = all_55000_steam[all_55000_steam['publishers'].isna() & all_55000_steam['developers'].notna()]
print(f"Number of observations with missing publishers present developers: {missing_publishers_present_developers.__len__()}")

# Checking where our developer is missing but the publisher is present
missing_developers_present_publishers = all_55000_steam[all_55000_steam['developers'].isna() & all_55000_steam['publishers'].notna()]
print(f"Number of observations with missing developers present publishers: {missing_developers_present_publishers.__len__()}")

# Checking where both our publisher and developer are missing
missing_developers_missing_publishers = all_55000_steam[all_55000_steam['developers'].isna() & all_55000_steam['publishers'].isna()]
print(f"Number of observations with missing developers and publishers: {missing_developers_missing_publishers.__len__()}")
    """, language='python')

    st.write(
        """
        * Percentage of games with the same publisher and developer: 67.61%
        * Number of observations with missing publishers present developers: 104
        * Number of observations with missing developers present publishers: 108
        * Number of observations with missing developers and publishers: 15
        """
    )

    st.write(
        """
        From some basic analytics, we can see we can impute $104 + 108 = 212$ observations using this method, $15$ will not be imputable.
        - It should be noted that categorical imputation cannot be done using many of the methods we were shown
            - Regression only works for numerical solution
            - KNN imputation does not make much sense in this context, though we may have nearest neighbors, our feature is somewhat unique to each entry no matter other numerical simmilarities.
                - SMOTE in a similar light.
        """
    )

    st.code(
        """
for i, row in missing_publishers_present_developers.iterrows():
    all_55000_steam.at[i, 'publishers'] = row['developers']
for i, row in missing_developers_present_publishers.iterrows():
    all_55000_steam.at[i, 'developers'] = row['publishers']        
        """, language='python'
    )

    st.dataframe(pd.DataFrame({'Null Counts': all_55000_steam.isnull().sum()}))

with meta_missing_tab:
    st.write(
        """
        With the metacritic reviews dataset, we have a the majority stake in missing values, with large proportions for many different features

        To analyze further, we perform missingness correlation analysis and qualitative missingness analysis using a heatmap:
        """
    )
    def plot_missingness_meta() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.2, 1]})

        # Visualizing missing values in a heatmap
        sns.heatmap(pre_metacritic.isnull().T, ax=axes[0], cbar=False)
        axes[0].set_title('Heatmap of Missing Values in Metacritic Data')

        missing_corr = pre_metacritic.drop(['platforms_info'], axis=1).isnull().corr() # Dropping platforms_info, since we have no missing values for platforms_info
        sns.heatmap(missing_corr, annot=True, cmap='coolwarm', vmax=1, vmin=-1, square=True, ax=axes[1])
        axes[1].set_title('Correlation of Missing Values Between Features')
        return fig
    st.pyplot(plot_missingness_meta())

    st.write(
        """
        Here we are drawing two heatmaps, one analyzing the feature missingness distribution against the observation indices, one analyzing feature missingness correlations, whether one feature being missing correlates to another also being missing.

        From our missingness distribution heatmap, we can notice that certain features seem to be missing in groups.

        -	Most notably, publisher and developer.
        -	Some observations also seem to be missing entirely.

        From our focused features missing values correlations, we can notice:

        -	genres missingness is 1:1 with name, meaning when name is missing genre is always missing.
        -	product_rating missingness has low correlation with the missingness of our other values
        -	user_score missingness is highly correlated (r = 0.91) with the missingness of user_ratings_count
            -	Intuitively, we can note that if we do not have any ratings, it would make sense that we would be lacking a score.
        """
    )

    st.dataframe(pre_metacritic.loc[pre_metacritic['name'].isnull()])

    st.write(
        """
        Looking closer at the few missing name values, we can notice that those entries that are missing name are also missing all their other feature columns. Since these observations do not contain any data, we can comfortably drop them. For further investigation, we may want to look into why these purely empty observations are within this dataset.
        """
    )
    st.code(
        """
metacritic = metacritic.dropna(subset=['name']) # Dropping observations only where name is missing
        """
    )

    st.write(
        """
        Performing a developer == publisher analysis yet again, we can notice that having a missing developer always correlates now to a missing publisher. They are mutually inclusive in missingness,
        we cannot perform the same imputation method as before.
        """
    )

    st.write(
        """
* Percentage of games with the same publisher and developer: 32.04%
* Number of observations with missing publishers present developers: 0
* Number of observations with missing developers present publishers: 0
* Number of observations with missing developers and publishers: 117
        """
    )


    st.write(
        """
        Games without a product rating are actually simply "Unrated", no committee has been used to diagnose the appropriate age range for this game. For this reason, we will impute a new value, "Unrated" for all missing values to better represent what the NaN originally was meant for.

        Though we can see that there are no "nulls" for platforms_info, upon further examination we can notice that several of our observations have empty lists for platforms_info. A total of 29 games within this last have empty lists, meaning no critic has reviewed the game for any platform.

        For the three game score rating related metrics, user_score, user_ratings_count and platforms_info, it would be inappropriate to impute without further examination.
        """
    )
    # - Dropping rows with missing values for essential features like `name` and `release_date`.
    # - Imputing missing values in numerical columns using median imputation.
    # - Keeping rows with missing values in certain categorical features to avoid losing potentially valuable information.


st.header("Joining Datasets")

st.write(
    """
    A short aside on merging datasets:

    Our initial intent was to create a conglomerate dataset using all three datasets. 
    - There are great features throughout our datasets, which we would like to see combined with the others.
    
    Through some preliminary analysis, we discovered the resulting merged size would be too small for meaningful analysis.
    """
)

st.code(
    """
top_1500_names = top_1500_steam['name'].str.lower().str.strip()
all_55000_names = all_55000_steam['name'].str.lower().str.strip()
metacritic_names = metacritic['name'].str.lower().str.strip()

print(type(top_1500_names))
common_names_nostrip = top_1500_steam['name'].isin(all_55000_steam['name'])
common_names = top_1500_names.isin(all_55000_names)

def get_common_names(df1, df2, standardize=False):
    if standardize:
        df1_names = df1['name'].str.lower().str.strip()
        df2_names = df2['name'].str.lower().str.strip()
        return df1_names.isin(df2_names).sum()
    else:
        return df1['name'].isin(df2['name']).sum()
print(get_common_names(top_1500_steam, all_55000_steam, standardize=True))
print(get_common_names(top_1500_steam, metacritic, standardize=True))
print(get_common_names(all_55000_steam, metacritic, standardize=True))
    """, language='python'
)
st.write(
    """
    * 113
    * 33
    * 413
    
    
    There are likely some confounding variables that cause our datasets to be quite mismatched, one postulation is that the dates of the snapshots are too far apart. Game marketplaces
    evolve too quickly, old games are cycled out, and newer games are cycled in.

    In any case, we proceed with our analysis on each dataset separately, however we try to bridge insights from one dataset to another where appropriate.
    """
)

st.header("Feature Engineering")

st.write(
    """
    Throughout every dataset, we very frequently utilized created features in our analysis. Having domain knowledge, we are able to postulate reasonable feature combinations for which further our
    analysis and answer more powerful questions. The feature engineering portions take part within the exploration of each branch.

    One common pattern of encoding we utilized was "binning and lumping", transforming continuous or discrete numerical features into categorical groupings.
    - For example, binning prices of games into types such as "Free to Play" or "Under $10".

    Binning and lumping helps us smooth the influence of outliers, we can place extreme outliers in outer bins, which reduces their impact on the rest of the data. Our datasets all deal with
    extreme outliers, this is just one of several methods we utilized to smooth them.
    """
)

# CONCLUSIONS
# st.header("Conclusions")

# st.write(
#     """
#     After merging and cleaning the datasets, the final dataframe used in the analysis contains:
#     - The `top_1500_steam` games dataset as the primary base.
#     - Additional review information from the `metacritic` dataset.
#     - Supplementary attributes from the `all_55000_steam` dataset.
    
#     This comprehensive dataset allows for a holistic analysis of the factors influencing game success, with attributes ranging from user reviews and ratings to revenue and release information.
#     """
# )