from IPython.display import display, HTML
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from time import sleep
from functools import wraps
import ast

# Borrowing tidier from previous assignment
def horizontal(dfs):
    html = '<div style="display:flex">'
    for df in dfs:
        html += '<div style="margin-right: 32px">'
        html += df.to_html()
        html += "</div>"
    html += "</div>"
    display(HTML(html))

def load_datasets() -> dict[str: pd.DataFrame]:
    top_1500_steam = pd.read_csv("data/top_1500_steam.csv")
    all_55000_steam = pd.read_csv("data/all_55000_steam.csv")
    metacritic = pd.read_csv("data/metacritic.csv")
    return {
        'top_1500_steam': top_1500_steam,
        'all_55000_steam': all_55000_steam,
        'metacritic': metacritic
    }

@st.cache_data
def load_top_1500_steam() -> pd.DataFrame:
    return pd.read_csv("data/top_1500_steam.csv")

@st.cache_data
def load_all_55000_steam() -> pd.DataFrame:
    return pd.read_csv("data/all_55000_steam.csv")

@st.cache_data
def load_metacritic() -> pd.DataFrame:
    return pd.read_csv("data/metacritic.csv")

def retry(max_retries=3, delay=1, backoff=2, logfile=None, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if logfile:
                        with open(logfile, 'a') as f:
                            f.write(f"{datetime.now()} - Error: {e}\n")
                            f.close()
                    retry_count += 1
                    if retry_count >= max_retries:
                        if logfile:
                            with open(logfile, 'a') as f:
                                f.write(f"{datetime.now()} - Max Retries exceeded: {e} returning None:\n")
                                f.close()
                        return None
                    sleep(delay * (backoff ** (retry_count - 1)))
        return wrapper
    return decorator

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
    
def literal_evaluate(platform_info_str):
    try:
        return ast.literal_eval(platform_info_str)
    except (ValueError, SyntaxError):
        return None  # Return None or a custom value if parsing fails