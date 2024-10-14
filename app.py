import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DEADMAN_PROJ",
    page_icon="🥴",
)
about = st.Page("about.py", title="About the Data", icon="💾")
developer_hypothesis = st.Page("developer_hypothesis.py", title="Page 3", icon="💀")
revenue_prediction = st.Page("revenue_prediction.py", title="Page 2", icon="💀")
documentation = st.Page("documentation.py", title="Documentation", icon="📔")

pg = st.navigation([about, revenue_prediction, developer_hypothesis, documentation])
pg.run()

# st.sidebar.success("Select a demo above.")

# st.title('_DVDMAN') # APP TITLE

# # LOAD
# data_load_state = st.text('Loading data...') 
# top_1500_steam = pd.read_csv("data/top_1500_steam.csv")
# data_load_state.text('Loading data...done!')

# st.subheader("RAW_DATA")
# st.write(top_1500_steam)

# st.subheader("TOP_1500_STEAM")
# fig = plt.figure(figsize=(10, 6))
# plt.hist(top_1500_steam['review_score'], bins=20, edgecolor='black')
# st.pyplot(fig)

# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
#     })

# option = st.selectbox(
#     'Which number do you like best?',
#     df['second column'])

# 'You selected: ', option