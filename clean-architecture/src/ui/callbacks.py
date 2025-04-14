# src/simulation_dashboard/ui/callbacks.py
import networkx as nx
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go


from src.use_cases import simulations
from src.infrastructure import visualizations
from src.domain.models import Grid, CascadingFailureSimulator
from src.ui import app



# Callback to render tab content
@app.callback(Output("tab-content", "children"),
              Input("tabs", "active_tab"))
def render_tab_content(active_tab):
    if active_tab == "tab-n1":
        return dbc.Container([
            html.H4("N-1 Contingency Analysis"),
            dbc.Button("Run N-1 Simulation", id="btn-n1", color="primary", className="my-2"),
            dcc.Graph(id="n1-graph")
        ])
    elif active_tab == "tab-renewable":
        return dbc.Container([
            html.H4("Renewable Energy Sensitivity Analysis"),
            dbc.Button("Run Renewable Simulation", id="btn-renewable", color="success", className="my-2"),
            dcc.Graph(id="renewable-graph")
        ])
    elif active_tab == "tab-maintenance":
        return dbc.Container([
            html.H4("Preventive Maintenance Simulation"),
            dbc.Button("Run Maintenance Simulation", id="btn-maintenance", color="warning", className="my-2"),
            dcc.Graph(id="maintenance-graph")
        ])
    elif active_tab == "tab-investment":
        return dbc.Container([
            html.H4("Infrastructure Investment & Capacity Planning"),
            dbc.Button("Run Investment Simulation", id="btn-investment", color="info", className="my-2"),
            dcc.Graph(id="investment-graph")
        ])
    elif active_tab == "tab-cyber":
        return dbc.Container([
            html.H4("Cyber-Physical System Security Simulation"),
            dbc.Button("Simulate Cyber Attack", id="btn-cyber", color="danger", className="my-2"),
            dcc.Graph(id="cyber-graph"),
            html.Div(id="cyber-eval", className="mt-2")
        ])
    elif active_tab == "tab-cascade":
        return dbc.Container([
            html.H4("Cascading Failure Simulation"),
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("Failure Probability"),
                        dbc.Input(id="input-prob", type="number", min=0, max=1, step=0.05, value=0.3)
                    ], className="mb-2")
                ], width=4),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("Steps"),
                        dbc.Input(id="input-steps", type="number", min=1, max=20, step=1, value=3)
                    ], className="mb-2")
                ], width=4),
                dbc.Col([
                    dbc.Button("Reset Cascade", id="btn-reset-cascade", color="primary", className="me-2"),
                    dbc.Button("Run One Step", id="btn-cascade-step", color="warning", className="me-2"),
                    dbc.Button("Run Multiple Steps", id="btn-cascade-multi", color="danger")
                ], width=4)
            ]),
            dcc.Graph(id="cascade-graph"),
            html.Div(id="cascade-eval", className="mt-2")
        ])
    return html.Div("Unknown Tab")

# Callback for N-1 simulation
@app.callback(Output("n1-graph", "figure"), Input("btn-n1", "n_clicks"))
def update_n1(n_clicks):
    if not n_clicks:
        return go.Figure()
    test_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
    ids, lost = simulations.run_n1_contingency(test_nodes)
    return visualizations.create_n1_chart(ids, lost)

# Callback for Renewable simulation
@app.callback(Output("renewable-graph", "figure"), Input("btn-renewable", "n_clicks"))
def update_renewable(n_clicks):
    if not n_clicks:
        return go.Figure()
    test_sources = {"Solar": 100, "Wind": 150, "Hydro": 80}
    ids, total, _ = simulations.run_renewable_sensitivity(test_sources, num_scenarios=50, seed=42)
    return visualizations.create_renewable_chart(ids, total)

# Callback for Maintenance simulation
@app.callback(Output("maintenance-graph", "figure"), Input("btn-maintenance", "n_clicks"))
def update_maintenance(n_clicks):
    if not n_clicks:
        return go.Figure()
    test_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
    scenarios = simulations.run_maintenance_simulation(test_nodes, max_outage=2)
    return visualizations.create_maintenance_chart(scenarios, threshold_fraction=0.80, baseline=sum(test_nodes))

# Callback for Investment simulation
@app.callback(Output("investment-graph", "figure"), Input("btn-investment", "n_clicks"))
def update_investment(n_clicks):
    if not n_clicks:
        return go.Figure()
    test_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
    scenarios = simulations.run_investment_simulation(test_nodes)
    return visualizations.create_investment_chart(scenarios)

# Callback for Cyber simulation
@app.callback(Output("cyber-graph", "figure"),
              Output("cyber-eval", "children"),
              Input("btn-cyber", "n_clicks"))
def update_cyber(n_clicks):
    if not n_clicks:
        return go.Figure(), ""
    grid = Grid()
    grid.simulate_attack(failed_nodes=["G1", "S2"], failed_edges=[("S2", "S3")])
    fig = visualizations.create_cyber_grid_figure(grid.graph, "Grid After Cyber Attack")
    eval_data = grid.evaluate()
    eval_text = f"Gen: {eval_data['generation']} MW, Demand: {eval_data['demand']} MW, Unserved: {eval_data['unserved']} MW"
    return fig, eval_text

# Global instance for cascading simulation.
cascade_sim = CascadingFailureSimulator(failure_probability=0.3)

@app.callback(Output("cascade-graph", "figure"),
              Output("cascade-eval", "children"),
              Input("btn-reset-cascade", "n_clicks"),
              State("input-prob", "value"),
              prevent_initial_call=True)
def reset_cascade(n_clicks, prob_value):
    cascade_sim.failure_probability = float(prob_value)
    cascade_sim.reset()
    fig = cascade_sim.get_network_figure("Cascading Failure Simulation (Reset)")
    eval_data = cascade_sim.evaluate()
    eval_text = f"Time: {cascade_sim.time_step} | Gen: {eval_data['generation']} MW, Unserved: {eval_data['unserved']} MW"
    return fig, eval_text

@app.callback(Output("cascade-graph", "figure"),
              Output("cascade-eval", "children"),
              Input("btn-cascade-step", "n_clicks"),
              prevent_initial_call=True)
def cascade_step(n_clicks):
    cascade_sim.cascade_step()
    fig = cascade_sim.get_network_figure(f"Cascading Failure (Step {cascade_sim.time_step})")
    eval_data = cascade_sim.evaluate()
    eval_text = f"Time: {cascade_sim.time_step} | Gen: {eval_data['generation']} MW, Unserved: {eval_data['unserved']} MW"
    return fig, eval_text

@app.callback(Output("cascade-graph", "figure"),
              Output("cascade-eval", "children"),
              Input("btn-cascade-multi", "n_clicks"),
              State("input-steps", "value"),
              prevent_initial_call=True)
def cascade_multi(n_clicks, steps):
    steps = int(steps)
    cascade_sim.run_cascade(steps=steps)
    fig = cascade_sim.get_network_figure(f"Cascading Failure (Step {cascade_sim.time_step})")
    eval_data = cascade_sim.evaluate()
    eval_text = f"Time: {cascade_sim.time_step} | Gen: {eval_data['generation']} MW, Unserved: {eval_data['unserved']} MW"
    return fig, eval_text
