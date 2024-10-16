import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_top_1500_steam

import plotly.graph_objects as go

top_1500_steam = load_top_1500_steam()
plt.style.use('ggplot')

st.header("🏆Exploring the Top Steam Games🏆")

st.write(
    """
    In this page, we will explore the top 1500 steam games dataset, a dataset containing the 1500 most profitable games on steam from this year (2024), up to Sept. 9th.
    The data was collected via the site "gamalytic.com". This page is divided into three main sections,
    """
)

st.markdown("""
1. [Features of Interest](#features-of-interest)
    * A preface on the unique features for this particular dataset, and some caveats to watch out for.
2. [Feature Explorations](#feature-explorations)
    * The exploration journey used to uncover key insights. Jumping to our dashboard is an option, but we _encourage_ a review of our exploration to really grasp understanding
    of our findings. 
    * This section introduces some basic statistics before branching into tabs for particular feature dives, allowing you to choose specific branches of exploration based on your interests.
3. [Key Insights](#key-insights)
    * A comprehensive dashboard showcasing what we deemed our most important findings, along with some caveats to consider.
""")

# Insert dashboards:
st.subheader("Features of Interest", divider=True)
st.write(
    """
    Before getting into the analysis, let's first explain our key features and how we intend to utilize them. We must also explain some various caveots regarding the 
    integrity of some particular features.

    For this particular dataset, we get a lot of great features to analyze that are in particular hard to source. The gamalytic API provides us with many useful data metrics that
    the base Steam API simply does not have. Here, we list a few specific highlighted features from this dataset for our analysis:
        
    - `copies_sold` and `revenue`:
        - Most datasets regarding game data do not have copies_sold or revenue listed, these figures are normally kept private. 
        This dataset also does not have exact figures$^1$, however we can trust that these figures may be more accurate than trivial 
        imputation methods, such as `price` times `copies_sold` for revenue, or something like review count times a metric for 
        `copies_sold`. gamalytic utilizes Steam Spy under the hood, allow it to probe and track steam user libraries and price changes over time.
        Using this extra information alongside traditional trivial methods, gamalytic imputes this information with higher degrees of accuracy.
    - `review_score`:
        - A numeric score metric representing the average of all user reviews for the game. For steam games, this metric in a numeric form is rarely given; 
        game review scores are normally ordinal labels, ("Negative", "Mixed", "Positive", etc.).
    - `publisher_class`:
        - Another metric rarely given in datasets, normally inferred from the combination of publisher(s) or developer(s) listed for a game.
    """
)

st.write(
    """
    ##### Some important caveats:
    1. The copies_sold and revenue features from Gamalytic are only estimates, the exact figures are kept private. 
    Gamalytic uses a complex algorithm for calculating both of these figures, see more here: [Link](https://gamalytic.com/blog/how-to-accurately-estimate-steam-sales)
    2. The original retail price varies and is not always the price calculated at sale. Discounts, price changes, and regional differences cause many confounding variables, thus 
    analysis using this metric should take these factors into consideration. 
    """
)

st.write("See our documentation page for further details on this dataset's features set:")

st.page_link("./documentation.py", label="Documentation", icon="📔")

st.write(
    """
    Using these particular features, we can try to analyze these questions:
    - Using review_score, we can ask: does the average review score influence revenue?
    - Using price, we can observe the distribution of price across the top games.
    - Using developer, we can observe the distribution of developers across the top games.
    - Using release_date, we can ask: how does the release_date affect revenue? Do more recent games have higher revenue?
    - Using publisher_class, we can ask: how does the publisher class distribution look for the top games? Do publisher classes correlate at all with revenue (e.g.
        do indie games earn less than large publishers?)
    """
)

st.subheader("Feature Explorations", divider=True)


st.write("#### General items")
st.write(
    """
        Basic statistics with feature listings:
    """
)

numerical_stats_tab, object_stats_tab = st.tabs(["Numerical Statistics", "Categorical Statistics"])

with numerical_stats_tab:
    st.write("### Numerical feature statistics")
    st.write(top_1500_steam.describe())

with object_stats_tab:
    st.write("### Categorical feature statistics")
    st.write(top_1500_steam[top_1500_steam.select_dtypes(include=['object', 'category']).columns].describe())

st.write(
    """
        Chart of publisher class distribution
    """
)
def plot_publisher_class_distribution() -> go.Figure:
    data = top_1500_steam['publisher_class'].value_counts().reset_index(name='count')

    # Create the pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=data["publisher_class"],
                values=data["count"],
                hoverinfo="label+percent",
                textinfo="value",
                textfont_size=14,
                marker=dict(line=dict(color="white", width=2)),
            )
        ]
    )
    fig.update_layout(
        title_text="Distribution of Publisher Classes",
        showlegend=True,
        height=400
    )
    return fig
st.plotly_chart(plot_publisher_class_distribution())
st.write(
    """
        These publisher classes are used in several analysis to answer questions regarding their differences.
    """
)

# ------------------------------------


sales_tab, release_date_tab, review_and_playtime_tab = st.tabs(["Sales Trends", "Timeline Insights", "Player Engagement Metrics"])

# ------------------------------------

with sales_tab:
    st.markdown("### Navigation")
    # TODO: Distribution Graphs for sales metrics
    def plot_sales_distributions() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        axes: list[plt.Axes] = axes.ravel()

        sns.histplot(
            data=top_1500_steam,
            x='copies_sold',
            bins=20,
            hue='publisher_class',
            multiple='stack',
            ax=axes[0]
        )

        sns.histplot(
            data=top_1500_steam,
            x='revenue',
            bins=20,
            hue='publisher_class',
            multiple='stack',
            ax=axes[1]
        )

        sns.histplot(
            data=top_1500_steam,
            x='price',
            bins=20,
            hue='publisher_class',
            multiple='stack',
            ax=axes[2]
        )

        return fig
    st.pyplot(plot_sales_distributions())

    st.write(
        """
        TODO: Comment
        """
    )

    def plot_top_n_copies_sold_and_revenue(copies_sold_n: int = 10, revenue_n: int = 10) -> plt.Figure:
        # Comparing copies sold within our top 10 sorted by copies sold
        fig, axes = plt.subplots(2, 1, figsize=((20, 12)))
        axes: list[plt.Axes]  = axes.ravel()

        sorted_by_copies_sold = top_1500_steam.sort_values(by='copies_sold', ascending=False)

        sns.barplot(
            x=sorted_by_copies_sold.head(copies_sold_n)["name"],
            y=sorted_by_copies_sold.head(copies_sold_n)["copies_sold"],
            ax=axes[0]
        )
        axes[0].set_title(f"Top {copies_sold_n} games by copies sold", fontsize=15, y=1.02)
        axes[0].set_xlabel("Title")
        axes[0].set_ylabel("Copies Sold (10 Million)")

        # Checking the proportion of copies sold from the top 10 to total copies sold of all games within our dataset:
        percentage_of_total_copies_sold_within_top_n = (sorted_by_copies_sold.head(copies_sold_n)['copies_sold'].sum() / top_1500_steam['copies_sold'].sum()) * 100
        print(f'Percentage of copies sold by top {copies_sold_n} to total copies sold within dataset: {percentage_of_total_copies_sold_within_top_n:.2f}%')

        # Comparing copies sold within our top 10 sorted by copies sold
        sorted_by_revenue = top_1500_steam.sort_values(by="revenue", ascending=False)

        sns.barplot(
            x=sorted_by_revenue.head(revenue_n)["name"],
            y=sorted_by_revenue.head(revenue_n)["revenue"],
            ax=axes[1]
        )
        axes[1].set_title(f"Top {revenue_n} games by revenue", fontsize=15, y=1.02)
        axes[1].set_xlabel("Title")
        axes[1].set_ylabel("Revenue (100 million)")

        # Checking the proportion of copies sold from the top 10 to total copies sold of all games within our dataset:
        percentage_of_total_revenue_within_top_n = (sorted_by_revenue.head(revenue_n)['revenue'].sum() / top_1500_steam['revenue'].sum()) * 100
        print(f'Percentage of revenue by top {revenue_n} to total revenue within dataset: {percentage_of_total_revenue_within_top_n:.2f}%')
        return fig
    st.pyplot(plot_top_n_copies_sold_and_revenue())

# ------------------------------------
with release_date_tab:
    st.markdown("### Navigation")



# ------------------------------------
with review_and_playtime_tab:
    st.markdown("### Navigation")


# ------------------------------------

st.write(
    """
        Starting with some distribution graphs:
    """
)

features_to_plot = ['copies_sold', 'price', 'revenue', 'avg_playtime', 'review_score', 'publisher_class']

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.ravel()
for i, feature in enumerate(features_to_plot):
    axes[i].hist(top_1500_steam[feature])
    axes[i].set_title(feature)

st.pyplot(fig=fig)

#TODO: Add some insights here
st.write(
    """
    Wow! Within the features copies_sold and revenue we can see a _heavy_ skew to the right. We have
    some extreme outliers within our dataset that are making the true distribution very hard to see.
    - This is somewhat suprising: though this dataset already deals with the top games on the market, 
    within the top games there are the _top_ top games.
    We have some more "normal" looking distributions for our price, and avg_playtime features.

    We can also notice that most of our games within our top games dataset are indie games!
    - The Indie game industry is a currently flourishing and highly contested sector of the global market. 
    Steam is a popular platform for indie developers to share their games, so this 'fact' is unsuprising. It
     may be instead useful to look at whether the concentration of indie games decreases when looking at the
     entire steam database, rather than just the top games.
    """
)

st.markdown("##### Basic Statistics")

#TODO: Add Basic Statistics
#TODO: Add Potential Insights from just basic statistics

st.markdown("##### Distribution Analysis")

#TODO: Move distribution analysis over to this section
#TODO: Add header anchors for different analysis within this section
#TODO: Create more distribution analysis for thinsg liek date

st.markdown("##### Correlation Analysis")

#TODO: Create a corrleation heatmap
#TODO: Add potential insights from the correlations shown

st.markdown("##### Feature Engineering")

#TODO: Create new features for analysis + Add some insights on choices made for feature engineering

st.markdown("##### Bivariate Analysis")

#TODO: Create regressions
#TODO: Analyse and create insights

st.markdown("##### Key Insights") # Plus Dashboards

#TODO: Conglomerate the key analysis into a dashboard and give conclusive closing remarks
#TODO: Include some possible metrics that would confound our results

#TODO Include a page that conglomerates the key insights from all three datasets into one conclusive analysis.
#TODO: Try to include reasoning behind why certain correlations may be present in one dataset that are not in another.

# Comparing copies sold within our top 10 sorted by copies sold
sorted_by_copies_sold = top_1500_steam.sort_values(by='copies_sold', ascending=False)
num_from_top = 10
plt.figure(figsize=(20, 6))
plt.bar(x = sorted_by_copies_sold.head(num_from_top)['name'], height= sorted_by_copies_sold.head(num_from_top)['copies_sold'])
plt.title(f"Top {num_from_top} games by copies sold", fontsize=15, y=1.02)
plt.xlabel("Title")
plt.ylabel("Copies Sold (10 Million)")
st.pyplot(fig=plt.gcf())

st.write(
    """
        Sorting by copies sold and looking at the top 10 games, we can get a better picture of the distribution within our outliers.

        By summing the total copies sold for the top 10 and dividing by the total copies sold within our dataset we can get a better idea of just how much our top games
        hog the marketplace.
    """         
)

# Checking the proportion of copies sold from the top 10 to total copies sold of all games within our dataset:
percentage_of_total_copies_sold_within_top_n = (sorted_by_copies_sold.head(num_from_top)['copies_sold'].sum() / top_1500_steam['copies_sold'].sum()) * 100
st.write(f'Percentage of copies sold by top {num_from_top} to total copies sold within dataset: {percentage_of_total_copies_sold_within_top_n:.2f}%')

st.write(
    """
        Within just our top 10, we are already accounting for _more than half_ of all game sales of the top 1500 games by revenue.

        The reality is, a small number of games seem to dominate the pack, capturing the majority of players.
    """         
)

start, stop = 100, 1000
selection = sorted_by_copies_sold.iloc[start:stop]
plt.figure(figsize=(20, 6))
plt.bar(x = list(range(start, stop)), height= selection['copies_sold']) # We do not really care about the names, just the distribution for now
plt.title(f"Rank {start} to {stop} games by copies sold", fontsize=15, y=1.02)
plt.xlabel("Rank")
plt.ylabel("Copies Sold")
st.pyplot(fig=plt.gcf())

st.write(
    """
        Disregarding our "top dogs", however, we can see that there is a long tail of games who still have a sizable stake in copies sold. It seems that, though there is a 
        heavy divide between our top games, and the rest, this divide does not truly reflect a "winner takes all" nature.
        - Though competition at the top may be extremely fierce, there is still a large market for games outside the top. We should try to investigate the marketplace further
         investigating in particular whether people who buy games in the top sales are also prone to buying games outside the top.
    """         
)

avg_copies_sold_by_publisher = top_1500_steam.groupby('publisher_class')['copies_sold'].mean().reset_index()
    
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(data=top_1500_steam, x='publisher_class', y='copies_sold', ax=axes[0])
axes[0].set_title('Boxplots for copies sold by publisher class')
axes[0].set_xlabel('Publisher Class')
axes[0].set_ylabel('Copies Sold')

sns.barplot(x=avg_copies_sold_by_publisher['publisher_class'], y=avg_copies_sold_by_publisher['copies_sold'], ax=axes[1])
axes[1].set_title('Average copies sold by publisher class')
axes[1].set_xlabel('Publisher Class')
axes[1].set_ylabel('Average Copies Sold')
st.pyplot(fig=fig)

st.write(
    """
        Disregarding our "top dogs", however, we can see that there is a long tail of games who still have a sizable stake in copies sold. It seems that, though there is a 
        heavy divide between our top games, and the rest, this divide does not truly reflect a "winner takes all" nature.
        - Though competition at the top may be extremely fierce, there is still a large market for games outside the top. We should try to investigate the marketplace further
         investigating in particular whether people who buy games in the top sales are also prone to buying games outside the top.
    """
)

st.markdown("##### Exploring Revenue")

# Comparing copies sold within our top 10 sorted by copies sold
sorted_by_revenue = top_1500_steam.sort_values(by='revenue', ascending=False)
num_from_top = 10
plt.figure(figsize=(20, 6))
plt.bar(x = sorted_by_revenue.head(num_from_top)['name'], height= sorted_by_revenue.head(num_from_top)['revenue'])
plt.title(f"Top {num_from_top} games by revenue", fontsize=15, y=1.02)
plt.xlabel("Title")
plt.ylabel("Revenue")
st.pyplot(fig=plt.gcf())

# Checking the proportion of copies sold from the top 10 to total copies sold of all games within our dataset:
percentage_of_total_revenue_within_top_n = (sorted_by_revenue.head(num_from_top)['revenue'].sum() / top_1500_steam['revenue'].sum()) * 100
st.write(f'Percentage of revenue by top {num_from_top} to total revenue within dataset: {percentage_of_total_revenue_within_top_n:.2f}%')

avg_revenue_by_publisher = top_1500_steam.groupby('publisher_class')['revenue'].mean().reset_index()
    
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(data=top_1500_steam, x='publisher_class', y='revenue', ax=axes[0])
axes[0].set_title('Boxplots for revenue by publisher class')
axes[0].set_xlabel('Publisher class')
axes[0].set_ylabel('Revenue')

sns.barplot(x=avg_revenue_by_publisher['publisher_class'], y=avg_revenue_by_publisher['revenue'], ax=axes[1])
axes[1].set_title('Average revenue by publisher class')
axes[1].set_xlabel('Publisher class')
axes[1].set_ylabel('Average revenue')
st.pyplot(fig=fig)

bins = [0, 1, 2, 4, 6, float('inf')]  # Define the bin edges
labels = ['No Prior Experience', '1', '2 to 3', '4 to 5', '6+']  # Define the labels for each bin

print(top_1500_steam[top_1500_steam['publisher_class'] == "AAA"]['publishers'].unique())

AAA_publisher_counts = top_1500_steam[top_1500_steam['publisher_class'] == "AAA"]['publishers'].value_counts().reset_index(name='Developed Games')
AAA_publisher_revenue = top_1500_steam[top_1500_steam['publisher_class'] == "AAA"].groupby('publishers')['revenue'].mean().reset_index()
AAA_publisher_summary = pd.merge(AAA_publisher_counts, AAA_publisher_revenue, left_on='publishers', right_on='publishers', how='left')

indie_publishers_exploded = top_1500_steam[top_1500_steam['publisher_class'] == "Indie"]['publishers'].str.split(',').explode()

indie_publisher_counts = indie_publishers_exploded.groupby(indie_publishers_exploded).size().reset_index(name='Developed Games')

indie_publisher_revenue = top_1500_steam[top_1500_steam['publisher_class'] == "Indie"].groupby('publishers')['revenue'].mean().reset_index()

indie_publisher_counts['experience'] = pd.cut(indie_publisher_counts['Developed Games'], bins=bins, labels=labels)

indie_publisher_summary = pd.merge(indie_publisher_counts, indie_publisher_revenue, left_on='publishers', right_on='publishers', how='left')

indie_publisher_summary.loc[indie_publisher_summary['revenue'].isna(), 'revenue'] = 0

indie_publisher_summary['publishers'] = indie_publisher_summary['publishers'].astype(str)

plt.figure()
sns.barplot(data=indie_publisher_summary, x='revenue', y='experience', hue='experience', orient='h')
plt.title("Average est. revenues with IQR by publisher experience")
st.pyplot(plt.gcf())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(data=AAA_publisher_counts, x='Developed Games', y='publishers', orient='h', ax=axes[0])
axes[0].set_title("AAA studios published games count")
sns.barplot(data=AAA_publisher_summary, x='revenue', y='publishers', orient='h', ax=axes[1])
axes[1].set_title("AAA studios average revenue per game")
plt.tight_layout()
st.pyplot(plt.gcf())

st.subheader("Key Insights", divider=True)

st.write(
    """

    """
)