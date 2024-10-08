from IPython.display import display, HTML

# Borrowing tidier from previous assignment
def horizontal(dfs):
    html = '<div style="display:flex">'
    for df in dfs:
        html += '<div style="margin-right: 32px">'
        html += df.to_html()
        html += "</div>"
    html += "</div>"
    display(HTML(html))