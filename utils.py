from IPython.display import display, HTML
import pandas as pd

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