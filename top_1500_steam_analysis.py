import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_top_1500_steam

import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

import calendar

top_1500_steam = load_top_1500_steam()
plt.style.use('ggplot')

def perform_feature_engineering():
    # Exploring price_category vs revenue and price vs copies sold
    bins = [-1, 0, 10, 50, float('inf')]  # -1 to include 0 in the first bin
    labels = ['Free to Play', 'Under $10', '$10-50', 'Over $50']
    top_1500_steam['price_category'] = pd.cut(top_1500_steam['price'], bins=bins, labels=labels)

    top_1500_steam['release_date'] = pd.to_datetime(top_1500_steam['release_date'], format='%Y-%m-%d')
    top_1500_steam['release_month'] = top_1500_steam['release_date'].dt.month_name()
    month_order = list(calendar.month_name[1:])
    top_1500_steam['release_month'] = pd.Categorical(top_1500_steam['release_month'], categories=month_order, ordered=True)

    bins = [-1, 10, 20, 40, 50, 70, 80, 95, float('inf')] # Based on how steam categorizes
    labels = ['Overwhelmingly Negative', 'Negative', 'Mostly Negative', 'Mixed', 'Mostly Positive', 'Positive', 'Very Positive', 'Overwhelmingly Positive']
    top_1500_steam['review_score_category'] = pd.cut(top_1500_steam['review_score'], bins=bins, labels=labels)
perform_feature_engineering()

st.header("🏆Exploring Top Steam Games🏆")

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
    * A comprehensive list of our most important findings, along with some caveats to consider.
""")

# Insert dashboards:
st.subheader("Features of Interest", divider=True)
st.write(
    """
    Before getting into the analysis, let's first explain our key features and how we intend to utilize them. We must also explain some various caveats regarding the 
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

def plot_correlation_heatmap() -> go.Figure:
    corr_matrix = top_1500_steam.select_dtypes(include='number').drop('steam_id', axis=1).corr()
    corr_matrix = np.round(corr_matrix, 2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True, 
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    fig.update_layout(
        title='Correlation heatmap',
        width=700,
        height=700,
        margin=dict(l=120, r=120, t=100, b=100)
    )
    
    return fig

pie_chart_tab, corr_heatmap_tab = st.tabs(['Publisher Class Distribution', 'Numerical Feature Correlations'])

with pie_chart_tab:
    st.plotly_chart(plot_publisher_class_distribution())
    st.write(
        """
            These publisher classes are used in several analysis to answer questions regarding their differences.
        """
    )

with corr_heatmap_tab:
    st.plotly_chart(plot_correlation_heatmap())
    st.write(
        """
            This heatmap shows the correlation between different numerical features. You might observe, besides the trivial revenue and copies_sold correlation,
            we have no feature pairs which correlate to each other very well! This influences our analysis away from simple regressions.
        """
    )

# ------------------------------------


st.write("### Branch Exploration")

sales_tab, release_date_tab, review_and_playtime_tab = st.tabs(["Sales Trends", "Timeline Insights", "Player Engagement Metrics"])

# ------------------------------------

with sales_tab:
    st.markdown("##### Sales Trends")

    st.write(
        """
            Within this branch, we want to investigate the following questions:
            - Is the market dominated by a few players, or is the market spread across many?
            - Does having a specific publisher for a game have an advantage in the market?
            - Does experience matter in terms of success? How does a seasoned developer fare against a new developer?
            - How much does price setting influence the sales success of games?
        """
    )

    @st.cache_data
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
        These distribution plot shows the distribution of copies sold, revenue, and price across the our top games.

        From this, we can observe some _massive_ outliers giving us a heavy skew to the right both in copies sold and revenue!
        - Evidently, from just our basic statistics we could have inferred this would happen, our standard deviation values are of higher magnitude than our average.
        
        Price seems much more "regular" in distribution.

        Let's have a closer look, now selecting on the top of the top games:
        """
    )

    @st.cache_data
    def plot_top_n_copies_sold_and_revenue(copies_sold_n: int = 10, revenue_n: int = 10) -> plt.Figure:
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

        sorted_by_revenue = top_1500_steam.sort_values(by="revenue", ascending=False)

        sns.barplot(
            x=sorted_by_revenue.head(revenue_n)["name"],
            y=sorted_by_revenue.head(revenue_n)["revenue"],
            ax=axes[1]
        )
        axes[1].set_title(f"Top {revenue_n} games by revenue", fontsize=15, y=1.02)
        axes[1].set_xlabel("Title")
        axes[1].set_ylabel("Revenue (100 million)")

        return fig
    st.pyplot(plot_top_n_copies_sold_and_revenue())

    st.write(
        """
        Within our top 10 in revenue and copies sold, the distribution looks suprisingly fairer. Though for both copies sold and revenue, we can see a clear favorite (who is suprisingly different),
        they are not magnitudes above the others within this set.

        With this in mind, lets take a look at how the rest of our top games look:
        """
    )

    @st.cache_data
    def plot_range_copies_sold_and_revenue(copies_sold_range: tuple[int, int] = (100, 1000), revenue_range: tuple[int, int] = (100, 1000)) -> plt.Figure:
        sorted_by_copies_sold = top_1500_steam.sort_values(by='copies_sold', ascending=False)
        sorted_by_revenue = top_1500_steam.sort_values(by="revenue", ascending=False)

        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        axes: list[plt.Axes] = axes.ravel()

        start, stop = copies_sold_range
        selection = sorted_by_copies_sold.iloc[start:stop]
        sns.barplot(
            x=list(range(start, stop)), 
            y=selection["copies_sold"],
            ax=axes[0]
        )
        axes[0].set_title(f"Rank {start} to {stop} games by copies sold", fontsize=15, y=1.02)
        axes[0].set_xticks([])
        axes[0].set_xlabel("Rank")
        axes[0].set_ylabel("Copies Sold")

        start, stop = revenue_range
        selection = sorted_by_revenue.iloc[start:stop]
        sns.barplot(
            x=list(range(start, stop)), 
            y=selection["revenue"],
            ax=axes[1]
        )
        axes[1].set_title(f"Rank {start} to {stop} games by revenue", fontsize=15, y=1.02)
        axes[1].set_xticks([])
        axes[1].set_xlabel("Rank")
        axes[1].set_ylabel("Revenue")
        return fig
    st.pyplot(plot_range_copies_sold_and_revenue())

    st.write(
        """
        Disregarding our "top dogs", we can see that there is a long tail of games who still have a sizable stake in copies sold. It seems that, though there is a 
        heavy divide between our top games, and the rest, this divide does not truly reflect a "winner takes all" nature.

        🌟Though competition at the top may be extremely fierce, there is still a large market for games outside the top. We should try to investigate the marketplace further
        investigating in particular whether people who buy games in the top sales are also prone to buying games outside the top.🌟

        Very interesting!

        Let's take a look at our second question now, does publisher class have a bearing on our sales success:
        """
    )

    @st.cache_data
    def plot_avg_copies_sold_and_revenue() -> plt.Figure:
        avg_copies_sold_by_publisher = top_1500_steam.groupby('publisher_class')['copies_sold'].mean().reset_index()
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes: plt.Axes = axes.ravel()

        sns.boxplot(data=top_1500_steam, x='publisher_class', y='copies_sold', ax=axes[0])
        axes[0].set_title('Boxplots for copies sold by publisher class')
        axes[0].set_xlabel('Publisher Class')
        axes[0].set_ylabel('Copies Sold')

        sns.barplot(x=avg_copies_sold_by_publisher['publisher_class'], y=avg_copies_sold_by_publisher['copies_sold'], ax=axes[1])
        axes[1].set_title('Average copies sold by publisher class')
        axes[1].set_xlabel('Publisher Class')
        axes[1].set_ylabel('Average Copies Sold')

        avg_revenue_by_publisher = top_1500_steam.groupby('publisher_class')['revenue'].mean().reset_index()
            
        sns.boxplot(data=top_1500_steam, x='publisher_class', y='revenue', ax=axes[2])
        axes[2].set_title('Boxplots for revenue by publisher class')
        axes[2].set_xlabel('Publisher class')
        axes[2].set_ylabel('Revenue')

        sns.barplot(x=avg_revenue_by_publisher['publisher_class'], y=avg_revenue_by_publisher['revenue'], ax=axes[3])
        axes[3].set_title('Average revenue by publisher class')
        axes[3].set_xlabel('Publisher class')
        axes[3].set_ylabel('Average revenue')
        return fig
    st.pyplot(plot_avg_copies_sold_and_revenue())

    st.write(
        """
        The plots on the left show boxplots for copies sold and revenue with separate boxplots for publisher classes. The plots on the right show averages for copies sold and revenue
        by publisher class.

        Evidently, yes! At least within our top games, we can see a bit of a divide. Our box plots are not as informative due to the sheer amount of outliers (outliers are indicated by the 'fliers' / circles), 
        but our grouped averages tell a different story.
        
        🌟Though the large majority of top games are indie games, AA and AAA publisher games beat indie games on average by quite a large margin.🌟
        - Thinking about this, intuitively this should make some sense. AAA and AA publishers are selective on their product's quality and have resources to promote and advertise games. 

        Let's take a look at our third question now, does developer experience have any bearing on our sales success:
        """
    )

    @st.cache_data
    def plot_experience_to_revenue():
        bins = [0, 1, 2, 3, 5, float('inf')]  # Define the bin edges
        labels = ['No Prior Experience', '1', '2', '3 - 4', '5+']  # Define the labels for each bin

        # ------------------------------------

        AAA_publisher_counts = top_1500_steam[top_1500_steam['publisher_class'] == "AAA"]['publishers'].value_counts().reset_index(name='Developed Games')
        AAA_publisher_revenue = top_1500_steam[top_1500_steam['publisher_class'] == "AAA"].groupby('publishers')['revenue'].mean().reset_index()
        AAA_publisher_summary = pd.merge(AAA_publisher_counts, AAA_publisher_revenue, left_on='publishers', right_on='publishers', how='left')
        AAA_publisher_counts['experience'] = pd.cut(AAA_publisher_counts['Developed Games'], bins=bins, labels=labels)
        AAA_publisher_summary = pd.merge(AAA_publisher_counts, AAA_publisher_revenue, left_on='publishers', right_on='publishers', how='left')
        AAA_publisher_summary.loc[AAA_publisher_summary['revenue'].isna(), 'revenue'] = 0
        AAA_publisher_summary['publishers'] = AAA_publisher_summary['publishers'].astype(str)


        indie_publishers_exploded = top_1500_steam[top_1500_steam['publisher_class'] == "Indie"]['publishers'].str.split(',').explode()
        indie_publisher_counts = indie_publishers_exploded.groupby(indie_publishers_exploded).size().reset_index(name='Developed Games')
        indie_publisher_revenue = top_1500_steam[top_1500_steam['publisher_class'] == "Indie"].groupby('publishers')['revenue'].mean().reset_index()

        indie_publisher_counts['experience'] = pd.cut(indie_publisher_counts['Developed Games'], bins=bins, labels=labels)
        indie_publisher_summary = pd.merge(indie_publisher_counts, indie_publisher_revenue, left_on='publishers', right_on='publishers', how='left')
        indie_publisher_summary.loc[indie_publisher_summary['revenue'].isna(), 'revenue'] = 0
        indie_publisher_summary['publishers'] = indie_publisher_summary['publishers'].astype(str)

        # ------------------------------------

        start, stop = (100, 500)

        bins = [0, 1, 2, 3, 5, float('inf')]  # Define the bin edges
        labels = ['No Prior Experience', '1', '2', '3 - 4', '5+']  # Define the labels for each bin

        developers_exploded = top_1500_steam['developers'].str.split(',').explode()
        developer_counts = developers_exploded.groupby(developers_exploded).size().reset_index(name='Developed Games')
        developer_revenue = top_1500_steam.groupby('developers')['revenue'].mean().reset_index()
        developer_counts['experience'] = pd.cut(developer_counts['Developed Games'], bins=bins, labels=labels)
        developer_summary = pd.merge(developer_counts, developer_revenue, left_on='developers', right_on='developers', how='left')
        developer_summary.loc[developer_summary['revenue'].isna(), 'revenue'] = 0
        developer_summary['developers'] = developer_summary['developers'].astype(str)

        sorted_by_revenue = developer_summary.sort_values(by='revenue', ascending=False)
        selection = sorted_by_revenue.iloc[start:]

        # ------------------------------------

        # display(selection)
        # display(indie_publisher_summary)
        # display(AAA_publisher_summary)

        fig_1, axes = plt.subplots(1, 2, figsize=(20 , 5))
        axes: list[plt.Axes] = axes.ravel()
        sns.barplot(data=selection, x='revenue', y='experience', hue='experience', orient='h', ax=axes[0])
        axes[0].set_title("Average est. revenues with IQR by developer experience")

        sns.barplot(data=indie_publisher_summary, x='revenue', y='experience', hue='experience', orient='h', ax=axes[1])
        axes[1].set_title("Average est. revenues with IQR by publisher experience")

        fig_2, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.barplot(data=AAA_publisher_counts, x='Developed Games', y='publishers', orient='h', ax=axes[0])
        axes[0].set_title("AAA studios published games count")
        sns.barplot(data=AAA_publisher_summary, x='revenue', y='publishers', orient='h', ax=axes[1])
        axes[1].set_title("AAA studios average revenue per game")
        plt.tight_layout()

        return fig_1, fig_2
    fig_1, fig_2 = plot_experience_to_revenue()
    st.pyplot(fig_1)

    st.write(
        """
        The plot on the left showcases average revenues of developers by their developer experience. The plot on the right showcases average revenues by publisher experience.
        - Both a developer and a publisher may theoretically get better with more experience, so it may be interesting to look at trends for both.
        - The plots also have an "errorbar", which indicates the inner quartile range for our revenues in each group. This helps us get an idea about the spread of our data.

        From our plot on developer experience to revenue, we can see an interesting trend. The first few games a developer creates seem to be their initially best ones, followed by a steep decline
        in their 3-4 game, and a sharp increase in success by their 5th and beyond.

        From our plot on publisher experience to revenue, we see that publishers with less games seem to fair better on average than publishers with more games.
        - This trend may be misleading:
        """
    )

    st.pyplot(fig_2)

    st.write(
        """
        Examining closer, looking specifically at our AAA games, we can see some recognizable IPs. "EA", "Ubisoft", "CAPCOM", etc., these are household names, it is no suprise that
        they are our most frequent AAA publishers within our top games.

        Interesting, however, when we look at the average revenue per game, those names seem to be less impactful. "EA", though they publish often, and their games reach the top, their games are not
        part of the _top_ top. This can be said for many of the AAA companies on this list.
        """
    )
    @st.cache_data
    def plot_copies_sold_and_revenue_regression():
        # Linear regression model
        X = top_1500_steam[['copies_sold']]
        y = top_1500_steam['revenue']
        model = LinearRegression().fit(X, y)
        predicted_revenue = model.predict(X)

        # Scatter plot with regression line
        fig = px.scatter(
            top_1500_steam,
            x="copies_sold",
            y="revenue",
            title="Copies Sold vs Revenue",
            labels={"copies_sold": "Copies Sold", "revenue": "Revenue"},
        )
        fig.add_traces(px.line(top_1500_steam, x="copies_sold", y=predicted_revenue).data)

        return fig
    st.plotly_chart(plot_copies_sold_and_revenue_regression())

    st.write(
        """
        Here we have plotted our copies sold against revenue, our strongest correlation from our correlation matrix.
        - Though this may be our strongest numerical correlation, this is not of much use to us. Naturally, this correlation is already very intuitively sound, more sales equating to more revenue.
        - What may be interesting to note is that we have an outlier at 30 million copies sold, but practically 0 revenue! We also have many many points that seem to have lots of sales but very little
        to show for that. 🌟Popular games are not neccessarily big earners all the time.🌟

        Let's take a look now at trends in our price metric. How does our success vary with price?
        """
    )

    @st.cache_data
    def plot_price_distribution() -> plt.Figure:
        plt.figure()
        sns.histplot(
            data=top_1500_steam,
            x='price',
            hue='publisher_class',
        )
        plt.title("Price distribution")
        plt.xlabel("Price")
        return plt.gcf()
    st.pyplot(plot_price_distribution())

    st.write(
        """
        This is our distribution graph for price colored by our publisher class. We will attempt to answer our question by first getting a good idea of how publisher class may influence our price metric.
        In particular, we should notice that we have spikes at certain prices. We seem to have spikes 0, 10, 15, 20, and a few minor ones from there. In this light, we are likely observing the "standard" prices
        that the game market takes.
        """
    )

    @st.cache_data
    def plot_price_category_distribution() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(
            data=top_1500_steam,
            x='price_category',
            hue='price_category',
            alpha=0.9,
            ax=axes[0]
        )
        axes[0].set_title("Price category distribution")
        axes[0].set_xlabel("Price category")

        sns.histplot(
            data=top_1500_steam,
            x='price_category',
            hue='publisher_class',
            multiple="fill",
            alpha=0.9,
            ax=axes[1]
        )
        axes[1].set_title("Price category distribution by publisher class")
        axes[1].set_xlabel("Price category")
        axes[1].set_ylabel("Percentage")

        return fig
    st.pyplot(plot_price_category_distribution())
    
    st.write(
        """
        Furthering on our basic distribution, we can lump our prices into categories to try and capture the "standardization" of price, ranging from "Free to Play" (0 cost) all the way to our most expensive.
        - Lumping in this fashion helps us account for some of our price outliers, but also somewhat dilutes the details, some certain spikes are missing now.

        Looking at our distribution lumped, it is much more clear what ranges are most popular, and by what publisher class. Indie games tend to take a lower price range, and AA and AAA tend to take 
        higher price ranges.
        - Interestingly, we can note that there is a somewhat significant portion of the "Free to Play" category taken by AAA publishers.
        """
    )

    def plot_price_and_revenue() -> plt.Figure:
        start, stop = (100, 1000)

        sorted_by_revenue = top_1500_steam.sort_values(by='revenue', ascending=False)

        selection = sorted_by_revenue.iloc[start:]

        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        axes: list[plt.Axes] = axes.ravel()

        sns.barplot(
            data=selection,
            x='price_category', 
            y='revenue',
            hue='price_category',
            dodge=False,
            estimator=np.mean, 
            errorbar=None,
            ax=axes[0]
        )
        axes[0].set_title('Average revenue by price category')
        axes[0].set_xlabel('Price category')
        axes[0].set_ylabel('Average revenue')

        sns.barplot(
            data=selection,
            x='price_category',
            y='revenue',
            hue='price_category',
            dodge=False,
            estimator=np.sum,
            errorbar=None,
            ax=axes[1]
        )
        axes[1].set_title('Total revenue by price category')
        axes[1].set_xlabel('Price category')
        axes[1].set_ylabel('Total revenue')
        return fig
    st.pyplot(plot_price_and_revenue())
    
    st.write(
        """
        Here we have plotted our average revenue per game by our price category, and total revenue earned within this dataset by games within our price categories.

        From this, we can see that evidently yes! Price seems to have a noticible effect on the success of our game.
        - 🌟Quite suprisingly, free to play games within our top games have a higher average revenue than our games priced below 10, and games priced between 10 and 50 dollars,
        and highly priced games seem to be most profitable on average.🌟
            - We should try and qualify this claim, however. We are selecting on the "top games" by revenue: this may not be reflective for all games, however it is interesting to be true even within our top games.
        - With the majority of games being within our $10 - 50 range, it is unsuprising that the total revenue is also most at that range.
        """
    )

plt.close()
# ------------------------------------
with release_date_tab:
    st.markdown("### Timeline Insights")

    st.write(
        """
        Within this branch, we want to investigate the following questions:
        - Are there any significant days which see a lot of game releases?
        - Does having a specific release date correlate at all with sales metrics?
        - How do our sales behave as our game ages?
        """
    )

    st.write(
        """
        To begin, we can get a good sense of whether any specific days are significant by simply plotting our distributions.
        """
    )

    def plot_release_date_distribution() -> tuple[plt.Figure, plt.Figure]:
        fig_1 = plt.figure(figsize=(15, 6))
        sns.histplot(
            data=top_1500_steam,
            x='release_date',
            hue='publisher_class',
            multiple='stack',
            kde=True,
            discrete=True,
        )
        plt.title("Release Date Distribution (Day)")

        fig_2 = plt.figure(figsize=(15, 6))
        sns.histplot(
            data=top_1500_steam,
            x='release_month',
            hue='publisher_class',
            multiple='stack',
            kde=True,
            discrete=True,
        )
        plt.title("Release Date Distribution (Month)")
        return fig_1, fig_2
    fig_1, fig_2 = plot_release_date_distribution()
    st.pyplot(fig_1)

    st.write(
        """
        Here we have plotted their our release date frequencies. We separate the releases by publisher class, but keeping our bars the same height by stacking.
        
        Evidently, there seems to be a lot of "noise". Releases seem to be sporadic, not even, in terms of when. Some days we see lots of releases, other days we see very few.
        - From this, we can gather that 🌟there are plenty of days to release your game when there are less other games being released.🌟
        - We have an interesting little dip around July of this year in terms of releases, it may be interesting to try and correlate this with some world events in that 
        month.
        """
    )

    st.pyplot(fig_2)

    st.write(
        """
        Instead of looking at any specific day, it may be more insightful to look at "release month" instead. Here we have plotted precisely that, also separated by publisher class.

        Evidently we have a dip in September, but this can be explained easily. Our dataset does not continue through Sept., it is a snapshot from earlier in September and thus less time was allowed for new games to release.
        - As a whole, however, it does seem that games releases were on the rise up until the summer months.

        Analysing with these release months, we can try to now get a metric for whether certain months have a better sales record for our top games.
        """
    )

    def plot_monthly_revenue_distribution() -> tuple[plt.Figure, plt.Figure]:
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        sns.boxplot(
            data=top_1500_steam,
            x='release_month',
            y='revenue',
            hue='release_month',
            dodge=False,
            showfliers=False,
            ax=axes[0]
        )
        axes[0].set_title('Revenue by release month (IQR)')
        axes[0].set_xlabel('Release month')
        axes[0].set_ylabel('Revenue (1 million)')

        sns.boxplot(
            data=top_1500_steam,
            x='release_month',
            y='copies_sold',
            hue='release_month',
            dodge=False,
            showfliers=False,
            ax=axes[1]
        )
        axes[1].set_title('Copies sold by release month (IQR)')
        axes[1].set_xlabel('Release month')
        axes[1].set_ylabel('Copies sold')
        return fig
    st.pyplot(plot_monthly_revenue_distribution())

    st.write(
        """
        Here we have plotted a series of boxplots for revenue against the release month.
        - To account for outliers, we are taking the inner quartile range (IQR) of revenues for each month.

        Evidently, there 🌟does not seem to be any particular month that has the best average revenue🌟, our averages from our boxplots seems to be pretty stable for both the revenue and copies sold.
        - It may be interesting to see whether this changes in the coming months!
        """
    )

    cutoff_date = pd.to_datetime('2024-09-09')
    revenue_per_day_since_release = top_1500_steam['revenue'] / (cutoff_date - top_1500_steam['release_date']).dt.days
    def plot_revenue_per_week() -> plt.Figure:
        fig = plt.figure(figsize=(15, 6))
        sns.boxplot(
            data = top_1500_steam,
            x = 'release_month',
            y = revenue_per_day_since_release * 30,
            hue='release_month',
            showfliers=False,
            dodge=False
        )
        plt.title('Average revenue of games per month since their release day')
        plt.xlabel('Month')
        plt.ylabel('Revenue per month')

        return fig
    st.pyplot(plot_revenue_per_week())

    st.write(
        """
        Although we don't have sales metrics for each day listed, we can "postulate" trivially a metric: the revenue of the game per month based on when it was released.

        From this, we can gather a general sense that, yes, games tend to earn their most money at the start of their release, but this metric is not entirely the full story.
        - Duration will always dilute the game revenue in some way, even if there are lots of sales later. We cannot get a true sense of this from our dataset without 
        gaining further information.
        """
    )

plt.close()
# ------------------------------------
with review_and_playtime_tab:
    st.markdown("### Player Engagement Metrics")

    st.write(
        """
        Within this branch, we want to investigate the following questions:
        - Do specific classes of publisher tend to get different review scores?
        - Does the review score tend to indicate success? Does high scores indicate that a game is successful in sales?
        """
    )

    st.write(
        """
        Starting with a distribution graph of our review scores:
        """
    )

    # Distribution of review score
    def plot_review_score_distribution() -> plt.Figure:
        fig = plt.figure(figsize=(10, 6))

        sns.histplot(
            data=top_1500_steam,
            x='review_score',
            bins=20,
        )
        plt.title("Review score distribution")
        plt.xlabel("Review score")
        return fig
    st.pyplot(plot_review_score_distribution())

    st.write(
        """
        Though we see a majority stake in higher review scores for our top games, we also see a large portion of games having a bottom of the barrel review score.
        - Not all top games are highly reviewed!

        Examining further now, differentiating by publisher class:
        """
    )

    def plot_review_score_distribution_with_publisher_class() -> plt.Figure:
        fig = plt.figure(figsize=(10, 6))

        sns.histplot(
            data=top_1500_steam,
            x='review_score',
            hue='publisher_class',
            bins=20,
        )
        plt.title("Review score distribution")
        plt.xlabel("Review score")
        return fig
    st.pyplot(plot_review_score_distribution_with_publisher_class())

    st.write(
        """
        We can see qualitatively that games from all publishers can achieve all ranks of review score.
        """
    )

    def plot_review_score_correlators() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes: list[plt.Axes] = axes.ravel()

        sns.scatterplot(
            data = top_1500_steam,
            x='review_score',
            y='revenue',
            ax=axes[0]
        )
        axes[0].set_title('Review score vs revenue')


        sns.scatterplot(
            data = top_1500_steam,
            x='review_score',
            y='copies_sold',
            ax=axes[1]
        )
        axes[1].set_title('Review score vs copies sold')
        return fig
    st.pyplot(plot_review_score_correlators())

    st.write(
        """
        From our scatter plots on review score vs revenue and review score vs copies sold, we can observe why we may not have good correlation between review score and these other numeric features.
        We have lots of games with low review scores, of 0, that actually make a sizeable amount of money!
        - This suggests that there may be other factors at play other than review score for commericial success.

        By ignoring the top contendors (outliers), we can examine this seemingly counterintuitive notion further:
        """
    )

    def plot_review_score_correlators_outside_top_n(start = 100) -> plt.Figure:
        sorted_by_revenue = top_1500_steam.sort_values(by="revenue", ascending=False)
        sorted_by_copies_sold = top_1500_steam.sort_values(by='copies_sold', ascending=False)

        selection_revenue = sorted_by_revenue.iloc[start:]
        selection_copies_sold = sorted_by_copies_sold.iloc[start:]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes: list[plt.Axes] = axes.ravel()

        sns.scatterplot(
            data = selection_revenue,
            x='review_score',
            y='revenue',
            ax=axes[0]
        )
        axes[0].set_title(f'Review score vs revenue (rank {start} and beyond)')


        sns.scatterplot(
            data = selection_copies_sold,
            x='review_score',
            y='copies_sold',
            ax=axes[1]
        )
        axes[1].set_title(f'Review score vs copies sold (rank {start} and beyond)')
        return fig
    st.pyplot(plot_review_score_correlators_outside_top_n())

    st.write(
        """
        Here we have the same relations plotted, however this time only with those games outside the top 100 for revenue and copies sold. With this, we can better see the problem of direct correlation
        with review score.

        🌟Evidently, there are many games with very high review score that see very little revenue, and there are games with very low review score that see very high revenue!🌟

        In an attemp to smooth this relation, we can bin together certain ranges of review score into review "categories".
        """
    )
    
    def plot_review_score_category() -> plt.Figure:
        fig = plt.figure(figsize=(15, 6))
        sns.barplot(
            data = top_1500_steam,
            x = 'review_score_category',
            y = 'revenue',
            hue = 'review_score_category',
            dodge = False,
            estimator=np.mean,
            errorbar=None
        )
        plt.title("Average revenue by review score category")
        return fig 
    st.pyplot(plot_review_score_category())

    st.write(
        """
        The binning method used matches directly with how Steam's categorical review score is. Their platform does not display numeric scoring for their games, but uses a ratio of positive to negative review counts to determine categories.
        - Overwhelmingly Negative: 0 - 10%
        - Negative: 10 - 20%
        - Mostly Negative: 20 - 40%
        - Mixed: 40 - 50%
        - Mostly Positive: 50 - 70%
        - Positive: 70 - 80%
        - Very Positive: 80 - 95%
        - Overwhelmingly Positive: 95 - 100%

        When we bin our review scores as percentages, we can actually see a general trend emerge! We actually do have some correlation within our top games.
        - Our seemingly low correlation was due to the existence of many "overwhelmingly negative" scored games having very great revenues.

        To check the power of this relation, lets draw a double regression line plot, modeling one regresion for scores of 0 - 10, and another for scores beyond that:
        """
    )

    def plot_review_score_double_regression() -> plt.Figure:
        sorted_by_revenue = top_1500_steam.sort_values(by="revenue", ascending=False)

        selection_revenue = sorted_by_revenue.iloc[100:]

        negative_review_scores = selection_revenue[selection_revenue['review_score'] <= 12]
        positive_review_scores = selection_revenue[selection_revenue['review_score'] > 9]

        negative_review_score_regression = LinearRegression()
        positive_review_score_regression = LinearRegression()

        x_negatives = negative_review_scores['review_score'].values.reshape(-1, 1)
        y_negatives = negative_review_scores['revenue'].values

        x_positives = positive_review_scores['review_score'].values.reshape(-1, 1)
        y_positives = positive_review_scores['revenue'].values

        negative_review_score_regression.fit(x_negatives, y_negatives)
        positive_review_score_regression.fit(x_positives, y_positives)

        y_predicted_negatives = negative_review_score_regression.predict(x_negatives)
        y_predicted_positives = positive_review_score_regression.predict(x_positives)

        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=selection_revenue,
            x='review_score',
            y='revenue',
            label='True Data',
            alpha=0.5,
        )

        plt.plot(
            negative_review_scores['review_score'],
            y_predicted_negatives,
            color='red',
            label='0 - 10 score regression',
        )

        plt.plot(
            positive_review_scores['review_score'],
            y_predicted_positives,
            color='green',
            label='10 - 100 score regression',
        )

        plt.xlabel('Review score')
        plt.ylabel('Revenue')
        plt.title('Review Score vs Revenue with Regression Lines')
        plt.legend()
        return fig
    st.pyplot(plot_review_score_double_regression())

    st.write(
        """
        Evidently, we end with the same conclusion. 🌟Review score is either not at all, or _even slightly negatively_ correlated with revenue within our top games.🌟
        - Our outliers in revenue cause our averages within each review score group to be skewed!

        We can clarify our first graph by excluding the top 100 ranks for revenue within our averages:
        """
    )

    def plot_review_score_category_top_n() -> plt.Figure:
        sorted_by_revenue = top_1500_steam.sort_values(by="revenue", ascending=False)

        selection_revenue = sorted_by_revenue.iloc[100:]
        fig = plt.figure(figsize=(15, 6))
        sns.barplot(
            data = selection_revenue,
            x = 'review_score_category',
            y = 'revenue',
            hue = 'review_score_category',
            dodge = False,
            estimator=np.mean,
            errorbar=None
        )
        plt.title("Average revenue by review score category (rank 100 and beyond)")
        return fig
    st.pyplot(plot_review_score_category_top_n())

    st.write(
        """
        By taking our top contendors our, we get a better picture of our true correlation. We actually can notice a _slight_ negative trend in averages!
        """
    )

plt.close()
# ------------------------------------

st.markdown("[Explore other branches](#branch-exploration)")

st.subheader("Key Insights", divider=True)

with st.container():
    st.markdown(
        """
        <style>
        .stContainer > div {
            width: 150%;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container(border=True):

        st.write(
            """
            **🌟Market concentration🌟**
            """
        )

        @st.cache_data()
        def dashboard_plot_cumulative_revenue() -> go.Figure:
            sorted_by_revenue = top_1500_steam.sort_values(by="revenue", ascending=False)
            cumulative_revenues = sorted_by_revenue['revenue'].cumsum()
            total_revenue = top_1500_steam['revenue'].sum()

            cumulative_revenues = pd.DataFrame(
                {
                    "rank": cumulative_revenues.index,
                    "cumulative_revenue": cumulative_revenues.values,
                    "proportion_to_total_revenue": cumulative_revenues.values / total_revenue,
                }
            )

            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=cumulative_revenues.index,
                y=cumulative_revenues['cumulative_revenue'],
                mode='lines',
                name='Cumulative revenue',
                hovertemplate=(
                    'Rank: %{x}<br>' +
                    'Cumulative revenue: %{y}<br>' +
                    'Proportion to total revenue: %{text:.4f}<br>' +
                    '<extra></extra>'
                ),
                text=cumulative_revenues['proportion_to_total_revenue']
            ))

            # Bringing attention to some key figures
            fig.add_annotation(
                x=10,
                y=cumulative_revenues.loc[9, 'cumulative_revenue'],
                text=f'Top 10 revenues accounts<br>for {cumulative_revenues.loc[9, 'proportion_to_total_revenue'] * 100:.2f}% of total',
                showarrow=True,
                arrowhead=2,
                ax = 130,
                ay = 30,
            )

            fig.add_annotation(
                x=100,
                y=cumulative_revenues.loc[99, 'cumulative_revenue'],
                text=f'Top 100 revenues accounts<br>for {cumulative_revenues.loc[99, 'proportion_to_total_revenue'] * 100:.2f}% of total',
                showarrow=True,
                arrowhead=2,
                ax = 80,
                ay = 40,
            )
            
            fig.update_layout(
                title="Cumulative revenues",
                xaxis_title="Rank",
                yaxis_title="Total revenue",
                width=1000,
                height=500
            )
            
            return fig
        st.plotly_chart(dashboard_plot_cumulative_revenue())

        st.write(
            """
            The top 10 games make up the majority of the revenue of all games.
            - This could be due to some few standout game releases of this year, such as "Pal World" and "Black Myth: Wukong".
            - Looking at the rest of the distribution, however, we notice that beyond the first 100, we have a long tail of
            games with similar success, which suggests that smaller games may still have a chance despite the dominating market.
            """
        )


    with st.container(border=True):

        st.write(
            """
            **🌟Publisher advantage🌟**
            """
        )

        @st.fragment()
        def dashboard_plot_publisher_class_revenues(use_median: bool = False, show_fliers: bool = False) -> go.Figure:
            dashboard_use_median_for_publisher_class_revenues = st.toggle("Use median revenue instead of mean")
            use_median = dashboard_use_median_for_publisher_class_revenues

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Boxplots for revenue by publisher class",
                    "Average revenue by publisher class"
                    if not use_median
                    else "Median revenue by publisher class",
                ),
            )
            
            publisher_classes = top_1500_steam['publisher_class'].unique()

            for publisher_class in publisher_classes:
                revenue_data = top_1500_steam[top_1500_steam['publisher_class'] == publisher_class]['revenue']
                
                q1 = revenue_data.quantile(0.25)
                q3 = revenue_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                if not show_fliers:
                    revenue_data = revenue_data[(revenue_data >= lower_bound) & (revenue_data <= upper_bound)]
                
                fig.add_trace(
                    go.Box(
                        y=revenue_data,
                        name=publisher_class,
                        boxmean=True,
                        boxpoints="all" if show_fliers else False
                    ),
                    row=1, col=1
                )
            
            if use_median:
                avg_revenue_by_publisher = top_1500_steam.groupby('publisher_class')['revenue'].median().reset_index()
            else:
                avg_revenue_by_publisher = top_1500_steam.groupby('publisher_class')['revenue'].mean().reset_index()

            fig.add_trace(
                go.Bar(
                    x=avg_revenue_by_publisher["publisher_class"],
                    y=avg_revenue_by_publisher["revenue"],
                    name="Average revenue" if not use_median else "Median revenue"
                ),
                row=1, col=2
            )

            fig.update_layout(
                title="Publisher class revenue analysis",
                width=1000,
                height=500,
            )

            fig.update_xaxes(title_text="Publisher class", row=1, col=1)
            fig.update_yaxes(title_text="Revenue", row=1, col=1)

            fig.update_xaxes(title_text="Publisher class", row=1, col=2)
            fig.update_yaxes(title_text="Average revenue" if not use_median else "Median revenue", row=1, col=2)

            st.plotly_chart(fig)
        dashboard_plot_publisher_class_revenues()

        st.write(
            """
            Publishing with a large company yields better average results
            - Larger publishers have access to larger budgets, which can give games better marketing and support for distribution. 
            This discrepancy could be explained by these kinds of factors.
            - In addition, publishers, like "EA" or "Nintendo" have a certain brand presence, people come to expect quality
            from established names in the industry which may attract more players and generate sales.
            """
        )

    with st.container(border=True):

        st.write(
            """
            **🌟Free Games Gain, Mid Range Games Reign🌟**
            """
        )

        @st.cache_data()
        def dashboard_plot_price_and_revenue() -> go.Figure:
            start, stop = (100, 1000)

            sorted_by_revenue = top_1500_steam.sort_values(by='revenue', ascending=False)

            selection = sorted_by_revenue.iloc[start:]

            avg_revenue_by_price = selection.groupby('price_category', observed=False)['revenue'].mean().reset_index()
            total_revenue_by_price = selection.groupby('price_category', observed=False)['revenue'].sum().reset_index()

            colors = pc.qualitative.Plotly

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Average revenue by price category", "Total revenue by price category"))

            fig.add_trace(
                go.Bar(
                    x=avg_revenue_by_price['price_category'],
                    y=avg_revenue_by_price['revenue'],
                    name='Average Revenue',
                    marker=dict(color=colors),
                    showlegend=False
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=total_revenue_by_price['price_category'],
                    y=total_revenue_by_price['revenue'],
                    name='Total Revenue',
                    marker=dict(color=colors),
                    showlegend=False
                ),
                row=1, col=2
            )

            fig.update_layout(
                title="Revenue by price category",
                width=1000,
                height=500,
            )

            fig.update_xaxes(title_text="Price category", row=1, col=1)
            fig.update_yaxes(title_text="Average revenue", row=1, col=1)

            fig.update_xaxes(title_text="Price category", row=1, col=2)
            fig.update_yaxes(title_text="Total revenue", row=1, col=2)

            return fig
        st.plotly_chart(dashboard_plot_price_and_revenue())
        
        st.write(
            """
            We have a mid-price domination in the total revenue of the marketplace, but higher priced and even free to play games typically earn more money on average than that price range.
            - Expensive games are also mostly sold by the larger publishers. This discrepancy could be explained by this fact; premium rates are associated with the higher production of a
            larger company, thus more attraction is had.
            - Free to play success suggests that there are other avenues of generating revenue that may even outperform the standard models.
            """
        )

    with st.container(border=True):

        st.write(
            """
            **🌟Release Day🌟**
            """
        )

        @st.cache_data()
        def plot_monthly_revenue_distribution() -> go.Figure:
            monthly_stats = top_1500_steam.groupby('release_month', observed=False)['revenue'].agg(['median', 'mean']).reset_index()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=monthly_stats['release_month'],
                    y=monthly_stats['median'],
                    mode='lines+markers',
                    name='Median revenue',
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=monthly_stats['release_month'],
                    y=monthly_stats['mean'],
                    mode='lines+markers',
                    name='Mean revenue',
                )
            )
            fig.update_layout(
                title="Average and median revenue by release month",
                xaxis_title="Release month",
                yaxis_title="Revenue (1 million)",
                width=1000,
                height=500
            )
            
            return fig
        st.plotly_chart(plot_monthly_revenue_distribution())

        st.write(
            """
            There seems to be no clear advantage for releasing a game in a certain month / season.
            - This could suggest that the demand for games is mostly steady throughout the year.
            - This also could be indicative of game industry artificial "staggered" release schedules [[11](https://videogamelaw.allard.ubc.ca/2019/09/11/eas-staggered-release-experiment/)][[12](https://sibs.llc/strategically-timing-your-game-release/)]. 
            Game companies tend to try to avoid releasing major titles around the timing of other companies to avoid overlapping hype and direct comparisons.
            """
        )

    with st.container(border=True):

        st.write(
            """
            **🌟Poorly Rated but Highly Profitable🌟**
            """
        )

        @st.fragment()
        def dashboard_plot_review_score_double_regression():
            regression_split_metric = st.slider("Review score to split regression on", 10, 90, value=15)

            sorted_by_revenue = top_1500_steam.sort_values(by="revenue", ascending=False)
            selection_revenue = sorted_by_revenue.iloc[100:].reset_index()

            abs_difference = abs(selection_revenue['review_score'] - regression_split_metric)
            closest_index = abs_difference.idxmin()
            closest_review_score = selection_revenue.iloc[closest_index]['review_score']

            negative_review_scores = selection_revenue[selection_revenue['review_score'] <= closest_review_score]
            positive_review_scores = selection_revenue[selection_revenue['review_score'] >= closest_review_score]

            negative_review_score_regression = LinearRegression()
            positive_review_score_regression = LinearRegression()

            x_negatives = negative_review_scores['review_score'].values.reshape(-1, 1)
            y_negatives = negative_review_scores['revenue'].values

            x_positives = positive_review_scores['review_score'].values.reshape(-1, 1)
            y_positives = positive_review_scores['revenue'].values

            negative_review_score_regression.fit(x_negatives, y_negatives)
            positive_review_score_regression.fit(x_positives, y_positives)

            y_predicted_negatives = negative_review_score_regression.predict(x_negatives)
            y_predicted_positives = positive_review_score_regression.predict(x_positives)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=selection_revenue["review_score"],
                    y=selection_revenue["revenue"],
                    mode="markers",
                    name="True Data",
                    marker=dict(opacity=0.5),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=negative_review_scores["review_score"],
                    y=y_predicted_negatives,
                    mode="lines",
                    name=f"0 - {regression_split_metric} score regression",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=positive_review_scores["review_score"],
                    y=y_predicted_positives,
                    mode="lines",
                    name=f"{regression_split_metric} - 100 score regression",
                )
            )

            fig.update_layout(
                title="Review score vs revenue with double regression line",
                xaxis_title="Review score",
                yaxis_title="Revenue",
                width=800,
                height=600,
                legend=dict(x=.8, y=1, bgcolor='rgba(255,255,255,0.3)')
            )

            # return fig
            st.plotly_chart(fig)
            
        dashboard_plot_review_score_double_regression()
        

        st.write(
            """
            Success does not seem related with game ratings, at least within our top games.
            - This suggests that commercial success may be influenced more by many external factors, such as publisher, brand, hype, or other metrics.
            - This also could indicate that if a game is sufficiently negative, it attracts its own type of crowd. Negative review scores may actually
            be appeal for some players in the market.
            """
        )
        
    # st.write(
    #     """
    #     From our qualitative analysis, we gathered these key insights:
        
    #     #### Regarding sales dynamics:
    #     - It is clear that the top games are themselves dominated by other top games. There are some key players in the market, that are hogs of sales metrics at the top.
    #         - Outside of this, however, there is still a sizable market, within our top 1500 most of our games are earning hundreds of thousands of dollars in revenue.
    #     - Though the games market is filled with a majority of indie games, AAA and AA publishing tends to correlate with higher revenue and sales than indie publishing
    #     - For the first few games, developers of top games tend to make around the same in sales per game, and evidently those with the most experience, 5+ games made, have the largest
    #     amount of success.
    #     - There seems to be positive correlation between price ranges and revenue, however there is also a huge market in "Free to Play" games, with free games earning more on average than
    #     games priced under 10 and between 10 and 50

    #     #### Regarding Time:
    #     - There is a lot of "noise" in the specific days of release. Some days see a lot of game releases while others see barely any at all.
    #     - There does not seem to be any specific release month or day that correlates heavily with sales.
    #     - Our sales per month or day naturally become diluted as our game ages.
    #         - Though this an obvious point, it is still worth reminding!
    #         - With more informative data, we may be able to better understand how a game's sales evolve over time. Our analysis is limited!

    #     These findings suggest that the game market does not have a specific preference for release timings.
    #     - There may be lots of confounding factors for this particular analysis making it difficult to isolate the impact of timing on sales
    #         - Different marketing and promotion campaigns might influence game sales regardless of release date.
    #         - High quality games might be selling well regardless of good timing. Since our list is of top contendors, this may be the case.

    #     #### Regarding review scores for games:
    #     - The lack of correlation between review score and revenue within our top games suggests that there may be other factors at play other than review to drive commercial success.
    #         - Effective marketing, for example, might make certain games profitable despite game quality. Though anecdotal, we can think of several games that fit this remark such as 
    #         Cyberpunk 2077 and Warcraft III Reforged.
    #         - It may also be that certain games have a "niche" audience. Some games may cater to specific audiences, where the general audience may find them overall lower in quality.

    #     Important Caveats 🔴:

    #     Survivorship Bias / Selecting on the Dependent Variable
    #     - For all conclusions and insights drawn for this dataset, we must factor in the fact that we are selecting only the **top** games in revenue for our analysis. We are also selecting on a pretty recently dataset!
    #     - Having a focus on these games means our analysis is limited to games to that already are successful, we cannot say that traits in these games are correlated to success simply because they are already successful, we do not know
    #     whether games that are unsuccessful share successful traits or not!
    #     """
    # )
