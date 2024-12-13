import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import load_datasets, load_steam_reviews
data = load_datasets()
# top_1500_steam, all_55000_steam, metacritic, steam_reviews = data['top_1500_steam'], data['all_55000_steam'], data['metacritic'], data['steam_reviews']
top_1500_steam, all_55000_steam, metacritic = data['top_1500_steam'], data['all_55000_steam'], data['metacritic']
steam_reviews = load_steam_reviews()

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
    - [Steam Reviews 2021](https://www.kaggle.com/datasets/najzeko/steam-reviews-2021)
    """
)

# Displaying each dataset with basic information
st.subheader("Dataset Previews", divider=True)

# Display the first few rows of each dataframe
top_1500_tab, all_55000_tab, metacritic_tab, steam_reviews_tab = st.tabs(["Top 1500 Steam Games", "All 55,000 Steam Games", "Metacritic Reviews", "Steam Reviews"])

with top_1500_tab:
    st.write("##### Top 1500 Steam Games Dataset")
    st.dataframe(top_1500_steam.head())
with all_55000_tab:
    st.write("##### All 55,000 Steam Games Dataset")
    st.dataframe(all_55000_steam.head())
with metacritic_tab:
    st.write("##### Metacritic Reviews Dataset")
    st.dataframe(metacritic.head())
with steam_reviews_tab:
    st.write("##### Steam Reviews Dataset")
    st.dataframe(steam_reviews.head())

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
show_basic_statistics(steam_reviews, "Steam Reviews Dataset")

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
    where neccessary.
    """
)

# MISSING DATA ANALYSIS
st.subheader("Missing Data Analysis", divider=True)

pre_top_1500_steam = pd.read_csv("data/pre_top_1500_steam.csv")
pre_all_55000_steam = pd.read_csv("data/pre_all_55000_steam.csv")
pre_metacritic = pd.read_csv("data/pre_metacritic.csv")
pre_steam_reviews = pd.read_csv("data/pre_steam_reviews.csv")

st.write(
    """
    Previews of our missingness:
    """
)

pre_top_1500_steam_nulls = pd.DataFrame({'Null Counts': pre_top_1500_steam.isnull().sum()})
pre_all_55000_steam_nulls = pd.DataFrame({'Null Counts': pre_all_55000_steam.isnull().sum()})
pre_metacritic_nulls = pd.DataFrame({'Null Counts': pre_metacritic.isnull().sum()})
pre_steam_reviews_nulls = pd.DataFrame({'Null Counts': pre_steam_reviews.isnull().sum()})

top_1500_steam_nulls = pd.DataFrame({'Null Counts': top_1500_steam.isnull().sum()})
all_55000_steam_nulls = pd.DataFrame({'Null Counts': all_55000_steam.isnull().sum()})
metacritic_nulls = pd.DataFrame({'Null Counts': metacritic.isnull().sum()})
steam_reviews_nulls = pd.DataFrame({'Null Counts': steam_reviews.isnull().sum()})

tab1, tab2 = st.tabs(['Unprocessed', 'Postprocessing'])
with tab1:
    col1, col2, col3= st.columns(3)
    with col1:
        st.write("Top 1500 Steam Games Missing Values Count")
        st.dataframe(pre_top_1500_steam_nulls)
        st.write("Steam Reviews Missing Values Count")
        st.dataframe(pre_steam_reviews_nulls)
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
        st.write("Steam Reviews Missing Values Count")
        st.dataframe(steam_reviews_nulls)
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
    - Within steam_reviews we have a small proportion of missing values in only two features, "review" and "author_playtime_at_review". Only a couple hundred out of hundreds of thousands.

    To determine how best to handle our missing data, we must performed extensive analysis. Within this
    next section, we showcase our exploration into the missingness, and handling of our missing data. Though you may skip this section and simply
    read our conclusions, we highly recommend reviewing this information to gain a better understanding of our decisions.
    """
)

st.subheader("Individual Missingness Handling", divider=True)

top_missing_tab, all_missing_tab, meta_missing_tab, steam_reviews_tab = st.tabs(["Top 1500 Steam Handling", "All 55000 Steam Handling", "Metacritic Handling", "Steam Reviews Handling"])

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

with steam_reviews_tab:
    st.write(
        """
        For our steam reviews dataset, we have a very low proportion of missing data in two categorical features "review" and 
        "author_playtime_at_review". Given the datatype of review being text data, it would be inappropriate to impute using numerical methods 
        (stochastic regression / SMOTE), and common categorical imputation methods like "by mode" or a modified kmeans make little sense. 
        The text of each review is likely to be very unique in content, dependent on the game reviewed and the author writing it.
        """
    )

    def plot_heatmaps_for_missingness_analysis_of_steam_reviews() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1.5, 1]})

        missing_data_indices = pre_steam_reviews[['review', 'author_playtime_at_review']].isnull().any(axis=1)
        filtered_df = pre_steam_reviews.loc[missing_data_indices, ['review', 'author_playtime_at_review']]
        filtered_df.reset_index(inplace=True, drop=True)

        sns.heatmap(filtered_df.isnull().T, cbar=False, cmap='viridis', ax=axes[0])
        axes[0].set_title('Heatmap of missing values for observations with missing values')

        missing_corr = filtered_df.isnull().corr()
        sns.heatmap(missing_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=axes[1])
        axes[1].set_title('Correlations in missing values')

        return fig
    
    st.pyplot(plot_heatmaps_for_missingness_analysis_of_steam_reviews())


    st.write(
        """
        Here we are drawing two heatmaps, one analyzing the feature missingness distribution against the observation indices, one analyzing feature missingness correlations, whether one feature being missing correlates to another also being missing.

        Filtering and plotting by null values of "review" and "author_playtime_at_review" reveals suprisingly that these two have missingness that seems to be negatively related. Specifically, wherever "review" is missing, "author_playtime_at_review" is not missing", and vice versa. This suggests a systematic relationship in the way these data points are missing.

        We can only postulate on why this relationship exists:
        - Perhaps the data collection process had some anomalies, for some observations it simply only recorded one of the two.
        - Perhaps some of the data was anomalous. Many things could go wrong with the data, for example a user who left a review perhaps later refunded their copy of the game. This might cause playtime to be missing. 
        - Users also might be leaving blank reviews, only recording whether they recommend the game or not.


        For our analysis, we will assume that missing reviews are legitimate reviews, but with no text data. In this situation, a user leaves a review only leaving their recommendation and playtime.
        - We will not make the same assumption for the author playtime missingness. Instead, we will utilize stochastic regression with our most reasonable correlators as features.
            - This is definitely overkill for our use case, however we would like to showcase some technical knowledge of advanced numerical imputation methods.   
        """
    )

    def plot_correlations_between_author_playtime_and_numerical_features() -> plt.Figure:
        steam_reviews['review_text_length'] = steam_reviews['review'].apply(lambda x: len(x)) # Feature engineering review text length to see if we can observe any connection between length and our target
        plt.figure(figsize=(12,8))
        correlators = steam_reviews.select_dtypes("number").drop(["app_id", "review_id", "author_steamid"], axis=1)
        sns.heatmap(
            correlators.corr()[['author_playtime_at_review']].T,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            cbar=False
        )
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.title("Correlations between author_playtime_at_review and numerical features")
        return plt.gcf()
    _ = plot_correlations_between_author_playtime_and_numerical_features()

    st.write(
        """
        From a truncated correlation heatmap, we can observe very few great correlators. For our regression model, we 
        utilize only the top 2 correlators: "author_playtime_forever" and "author_playtime_last_two_weeks".
        
        From here, we trained a linear regression model on these two correlators, using an 80% train test split, then calculated our fitting metrics of mean absolute
        error and mean squared error.
        """
    )

    with st.expander("Reveal code block"):
        code = '''
            steam_reviews_not_missing = steam_reviews[steam_reviews['author_playtime_at_review'].notnull()]
            steam_reviews_with_missing = steam_reviews[steam_reviews['author_playtime_at_review'].isnull()]
            predictors = ['author_playtime_forever', 'author_playtime_last_two_weeks']
            target = 'author_playtime_at_review'

            def create_and_validate_linear_regression_model(data: pd.DataFrame, predictors: list[str], target: str, handle_outliers: bool) -> list[LinearRegression, list[float]]:
                X = data[predictors] # Design
                y = data[target] # Target

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=1
                )

                # Perform outlier handling
                if handle_outliers:
                    train_data = pd.concat([X_train, y_train], axis=1)
                    z_scores = np.abs(zscore(train_data))

                    # Filter out the outliers by zscore with an absolute threshold of 3
                    threshold = 3
                    train_data_filtered = train_data[(z_scores < threshold).all(axis=1)]
                    
                    X_train = train_data_filtered[predictors]
                    y_train = train_data_filtered[target]

                model = LinearRegression()
                model.fit(X_train, y_train)

                # Validation
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                print(f"Mean absolute error: {mae}")
                print(f"Mean squared error: {mse}")
                
                # Quick check against a mean baseline model
                mean_baseline = y_train.mean()
                y_baseline_pred = np.full_like(y_test, fill_value=mean_baseline)
                baseline_mae = mean_absolute_error(y_test, y_baseline_pred)
                baseline_mse = mean_squared_error(y_test, y_baseline_pred)
                print(f"Baseline (impute by mean) mean absolute error: {baseline_mae}")
                print(f"Baseline (impute by mean) mean squared error: {baseline_mse}")

                training_residuals = y_train - model.predict(X_train) # Used later for magnitude of stochastic noise
                return model, training_residuals

            _, _ = create_and_validate_linear_regression_model(
                steam_reviews_not_missing, predictors, target, False
            )
        '''
        st.code(code, language='python')
    
    st.write(
        """
        - Further testing reveals that adding more correlators only serves to reduce our accuracy in terms of mean squared error and mean absolute error. Sticking to our top 2 gives us our best results.
        
        Our results are as follows:
        """
    )

    st.markdown(
        """
        - `Mean absolute error: 4615.606251597345`
        - `Mean squared error: 191888428.36573023`
        - `Baseline (impute by mean) mean absolute error: 11340.927833778102`
        - `Baseline (impute by mean) mean squared error: 716729755.876812`
        """
    )
    st.write(
        """
        Even with our best predictors, we come up quite short in the accuracy department. From our basic statistical analysis, we can 
        notice that all numerical values regarding playtime are highly skewed to the right. Our results may be thrown by the presence of extreme 
        outliers. For this reason, we run the same test again, but first filtering for outliers in our training data.
        - By only filtering for outliers in our training data, we avoid overly optimistic performances. Outliers, after all, are legitimate and valid data 
        points. The actual missing data values are likely to have variability that is closer to the actual data before filtering.
        """
    )

    st.markdown(
        """
        - `Mean absolute error: 4616.919491273922`
        - `Mean squared error: 195409099.31966552`
        - `Baseline (impute by mean) mean absolute error: 9872.670436856642`
        - `Baseline (impute by mean) mean squared error: 724954230.9760497`
        """
    )

    st.write(
        """
        Even after accounting for outliers (absolute z-score > 3) we find that our mean squared error does not improve at all. 
        Evidently, our author playtime data may be too spread to generalize accurately with linear models. Comparing the performance of our 
        model against a simple baseline metric such as imputing just by mean shows that we at least manage to beat the baseline by a good margin.

        For this reason, we continue with our more accurate method, now adding the "stochastic" part in, 
        reintroducing variation into our predictions to impute with variation.
        """
    )

    with st.expander("Reveal code block"):
        code = '''
        # Adds normally distributed noise into a set of values
        def add_stochastic_noise(predictions, std_deviation):
            noise = np.random.normal(0, std_deviation, size=predictions.shape)
            return np.abs(predictions + noise)

        # Predictions
        y_missing_pred = model.predict(steam_reviews_with_missing[predictors])
        y_missing_pred_stochastic = add_stochastic_noise(y_missing_pred, residuals.std())

        # Impute
        steam_reviews.loc[steam_reviews_with_missing.index, 'author_playtime_at_review'] = y_missing_pred_stochastic
        '''
        st.code(code, language='python')

    st.write("After imputation, we examine the effect imputation has on our numerical statistics for our target.")

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pre_steam_reviews.describe()[['author_playtime_at_review']])
    with col2:
        st.dataframe(steam_reviews.describe()[['author_playtime_at_review']])

    st.write("We see that we have very little change, which informs us in how much integrity we keep.")

st.divider()

st.write("""
    Some general notes and caveats:
    - Due to the data type of our missing data being categorical, certain methods for handling missing values, such as stochastic regression or SMOTE, are unapplicable. 
    Our categorical values are very unique to each observation, and we cannot reasonably impute them using values from other observations.
    - Furthermore, simple approaches to categorical missingness handling, such as imputing by mode, would only serve to damage the integrity of our analysis.
         - Features such as "publisher" and "developer" are key in our analysis. Imputing by frequency would ruin the integrity of the true uniqueness of these values.
""")

st.header("Joining Datasets", divider=True)

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
    `113`
    `33`
    `413`
    
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

    Some common patterns we used:
    - "Binning and lumping": transforming continuous or discrete numerical features into categorical groupings.
        - For example, binning prices of games into types such as "Free to Play" or "Under $10".
        - Helps us smooth the influence of outliers and relate continuous data to relatable breakpoints. In addition, this
        also helps cuts the noise of the data down.
        - The biggest advantage may be in a simpler interpretation, however, facilitating much easier trend analysis and giving more 
        targeted decision-making.
    - "Explosion": Taking a list feature and exploding it into binary categorical columns.
        - For example, taking a list of genres and exploding into separate columns to allow multiple genres per
        observation.
        - Used a lot for summing and counting list format categorical data.
    - "Counting": Grouping by a categorical feature and counting occurances. Lets us convert unruly categorical data into a numerical format.
        - Sometimes used with explosion, for example counting how many games a certain developer has worked on.
    
    By far, binning and lumping was used the most. Outside these common patterns, we often did more unorthodox engineering to cater to very
    particular questions.

    Here we list a comprehensive list of our engineered features, by dataset:
    """
)

(
    top_1500_engineering_tab,
    all_55000_engineering_tab,
    meta_engineering_tab,
    steam_reviews_engineering_tab,
) = st.tabs(
    [
        "Top 1500 Steam Engineering",
        "All 55000 Steam Engineering",
        "Metacritic Engineering",
        "Steam Reviews Engineering",
    ]
)
with top_1500_engineering_tab:
    st.write(
        """
        - `price_category`:
            - Binning `price` to industry standard ranges, allowing for better comparison and clearer decision making.
        - `release_month`:
            - Binning `release_date` by month. Helps cut noise by day and give more notice to larger patterns.
        - `review_score_category`:
            - Binning `review_score` to Steam terminology. Steam uses category rather than numerical score, binning to this format gives 
            real world relatability to analysis.
        """
    )

with all_55000_engineering_tab:
    st.write(
        """
        - `price_category`:
            - Binning `price` to industry standard ranges, allowing for better comparison and clearer decision making.
        - `total_review_count`:
            - Sum of `positive_reviews` and `negative_reviews`.
        - `positive_ratio`:
            - Ratio of `positive_reviews` to `negative_reviews`. A ratio between rather than a ratio from total gives account to games with few reviews.
        - `total_review_bins`:
            - Binned `total_review_count` to reduce the skew of the data. Past a certain amount of reviews we care little about the variance.
        - `release_year`:
            - Binning `release_date` by year. Helps cut noise by day and give more notice to larger patterns.
        - `developer_experience` and `publisher_experience`:
            - Total games developed by the developer(s) or published by the publisher(s).
        - `[genre]`:
            - Exploded binary feature representing whether a game is associated with the genre or not. There are 30 different unique exploded genre columns.
            - Used in machine learning modeling analysis to help answer whether genre has an effect on success.
        - `[language]`:
            - Exploded binary feature representing whether a game has support for a particular language. There are over 50 different unique language columns.
            - Used in machine learning modeling analysis as an accounted confounder. Language may be an enabler for players to
            become at all interested in a particular game.
        - `languages_supported`:
            - A count of the number of languages a game supports.
            - Used in machine learning modeling analysis as an accounted confounder.
        """
    )

with meta_engineering_tab:
    st.write(
        """
        - `platform_name`, `platform_metascore`, `platform_metascore_count`:
            - Extracting information from the `platforms_info` feature, the name and scores given.
            - This is crucial information regarding game 'critic' review, allowing for analysis against user review.
        - `platform_frequency`:
            - Count on the frequency of the platform relative to the dataset total observation length.
        """
    )

with steam_reviews_engineering_tab:
    st.write(
        """
        - `review_text_length`, `review_total_words` and `review_total_sentences`:
            - Features that quantify length of review in different ways. Having different scales allows for different analysis,
            smaller scales reveal more granular patterns.
        - `even_cleaner_tokenized_review`:
            - After cleaning the `review` text with lowercasing, link removal and alphanumeric filtering (no symbols), we
            then tokenize and clean further (hence the 'even' cleaner). We follow by expanding contractions, removing stopwords and lemmatizing.
            - Note: the stopwords list is custom, an advancement from NLTK's library that includes more works like "like" and "really".
            Here is a [link](https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt). 
        """
    )

st.divider()

st.write(
    """
        In our analysis pages you may watch out for where these engineered features are used for direct relevance of their
        usages. For any analysis which use features not listed here, those engineered features can be assumed to be ephemeral, only used once and
        not saved directly back into the dataframe.

        Please see our codebase for more information on the exact details of the engineering. These details can be found in the individual exploration jupyter notebook files.
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