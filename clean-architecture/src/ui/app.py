# src/simulation_dashboard/ui/app.py
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Multi-Use-Case Simulation Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Multi-Use-Case Simulation Dashboard"), width=12)
    ], className="my-2"),
    dbc.Tabs([
        dbc.Tab(label="N-1 Contingency", tab_id="tab-n1"),
        dbc.Tab(label="Renewable Sensitivity", tab_id="tab-renewable"),
        dbc.Tab(label="Preventive Maintenance", tab_id="tab-maintenance"),
        dbc.Tab(label="Investment Planning", tab_id="tab-investment"),
        dbc.Tab(label="Cyber-Physical Security", tab_id="tab-cyber"),
        dbc.Tab(label="Cascading Failures", tab_id="tab-cascade")
    ], id="tabs", active_tab="tab-n1"),
    html.Div(id="tab-content", className="p-4")
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
