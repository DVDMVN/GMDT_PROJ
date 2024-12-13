# Game Market Insights Dashboard
<table>
<tr>
<td>
  A webapp made with Streamlit to display analysis of several game market datasets. Our goal is to assist game developers in making data-informed decisions by showcasing key factors that influence game success.
</td>
</tr>
</table>


## Demo
Check our live demo!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gmdtproj.streamlit.app/)

## Application Structure

This application is structured in an "article" like fashion. This is to mimic what game developers may typically use to gather information, through articles and newsletters. Users may navigate to 'pages', which are fashion in a 'medium' article sort of fashion. Each page is self contained, having information regarding the structure of the page, comprehensive pathing for the page. Much like a quest, we suggest that readers go through our exploration, however players that find the exploration waning may skip straight to the "boss", our key insights gathered from each dataset.

## File Explanation:

**Exploration**:

Initial IDA and EDA as well as cleaning and preprocessing was largely performed using Jupyter Notebooks.
- cleaning.ipynb
  - This notebook handles loading and cleaning of the raw datasets.
- exploring[*].ipynb
  - Notebooks which hold our experiments in exploratory data analysis for each dataset.

**Data**:

The 'data' folder is used to hold our preprocessed data. You may download and load the raw datasets according to cells in the 'cleaning.ipynb' file.
- We also hold the 'unprocessed' versions, of the data. This data is at the in between stage of preprocessing and postprocessing, standardized but not cleaned. Used in the 'about.py' file for demonstrations in cleaning.
- A special case, we also hold the 'engineered' version of the 'steam_reviews' data. Due to the size and nature of the dataset, performing feature engineering takes a considerable amount of time, even in low proportions. Holding the already feature engineered version saves on execution times.

**Streamlit Application**:

This streamlit application is split into different files, representative of their page.
- app.py
  - The head file of our application. 
- welcome.py
  - The starting page, holding basic orientation information.
- about.py
  - Basic overviews of the dataset, and some demonstrations and explanations of the cleaning process.
- top_1500_steam_analysis.py
  - Analysis of key features within the 'top_1500_steam.csv' data file.
- all_55000_steam_analysis.py
  - Analysis of key features within the 'all_55000_steam.csv' data file.
- metacritic_analysis.py
  - Analysis of key features within the 'metacritic.csv' data file.
- steam_reviews_analysis.py
  - Analysis of key features within the 'steam_reviews.csv' data file.
- documentation.py
  - An appendix.

**Extra Files**:

Some extra files were used to assist the main pages or exploration.
- utils.py
  - Storage for some common functions and other utilities. Holds our Steam API polling functions.
- steam_api_logging.txt
  - Tracks successes and errors from utilizing the Steam API.
- floppy_disc.ipynb
  - Utility notebook for quick problem troubleshooting.

### Development

Currently, this project is not open to contribution. This project was made for academics.

## Team
[![DVDMAN](https://avatars.githubusercontent.com/u/183556656?v=4&s=144)](https://github.com/DVDMVN)
