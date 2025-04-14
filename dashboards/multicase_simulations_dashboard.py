#!/usr/bin/env python3
"""
Multi-Use-Case Simulation Dashboard

This script builds a multi-tab simulation dashboard for six use cases:
1. N‑1 Contingency Analysis
2. Renewable Energy Integration & Sensitivity Studies
3. Preventive Maintenance & Reliability Engineering
4. Infrastructure Investment & Capacity Planning
5. Cyber-Physical System Security
6. Cascading Failure Simulation with an interactive dashboard interface

Run this script to launch the dashboard.
"""

import random
import logging
import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px

# Dash-related imports
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Configure logging globally
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


######################################
# Simulation Functions & Classes
######################################

# -------------------------------
# 1. N-1 Contingency Analysis
# -------------------------------
def simulate_n1_contingency(nodes):
    """
    Simulate N-1 contingency analysis.
    Given a list of production capacities, simulate the removal of each one.
    Returns:
        scenario_ids: list of node indices,
        lost_ratio: list of lost production ratios (in %).
    """
    nodes = np.array(nodes, dtype=float)
    total_production = nodes.sum()
    scenario_ids = list(range(len(nodes)))
    lost_ratio = [(node / total_production) * 100 for node in nodes]
    return scenario_ids, lost_ratio

def create_n1_bar_chart(scenario_ids, lost_ratio):
    fig = px.bar(x=scenario_ids, y=lost_ratio,
                 labels={'x': 'Node Index', 'y': 'Lost Production (%)'},
                 title="N-1 Contingency Analysis")
    fig.update_layout(yaxis_range=[0, max(lost_ratio)*1.2])
    return fig

# -------------------------------
# 2. Renewable Energy Integration & Sensitivity
# -------------------------------
def simulate_renewable_sensitivity(sources, num_scenarios=50, seed=None):
    """
    Simulate sensitivity analysis for renewables.
    sources: dict (e.g., {"Solar":100, "Wind":150, "Hydro":80})
    Returns:
        scenario_ids: list of scenario indexes,
        total_production: list of total production per scenario,
        multipliers: list of dicts with source multipliers.
    """
    if seed is not None:
        np.random.seed(seed)
    scenario_ids = list(range(num_scenarios))
    total_production = []
    multipliers = []
    for s in scenario_ids:
        scenario_factors = {}
        prod = 0.0
        for src, cap in sources.items():
            factor = np.random.uniform(0.2, 1.0)
            scenario_factors[src] = factor
            prod += cap * factor
        total_production.append(prod)
        multipliers.append(scenario_factors)
    return scenario_ids, total_production, multipliers

def create_renewable_scatter(scenario_ids, total_production):
    fig = px.scatter(x=scenario_ids, y=total_production,
                     labels={'x': 'Scenario ID', 'y': 'Total Production (MW)'},
                     title="Renewable Sensitivity Analysis")
    return fig

# -------------------------------
# 3. Preventive Maintenance & Reliability
# -------------------------------
from itertools import combinations
def simulate_maintenance(nodes, max_outage=2):
    """
    Simulate maintenance scenarios.
    For each combination of outage of 1 up to max_outage nodes.
    Returns a list of dictionaries with scenario details.
    """
    scenarios = []
    total_prod = sum(nodes)
    # Single and dual outages.
    for outage_count in range(1, max_outage+1):
        for off in combinations(range(len(nodes)), outage_count):
            offline = sum([nodes[i] for i in off])
            remaining = total_prod - offline
            loss_ratio = (offline/total_prod)*100
            scenarios.append({
                "scenario": str(off),
                "offline": offline,
                "remaining": remaining,
                "loss_ratio": loss_ratio
            })
    return scenarios

def create_maintenance_bar_chart(scenarios, threshold_fraction=0.80, baseline=None):
    """
    Create a bar chart for maintenance simulation.
    Threshold is set as a fraction of baseline production.
    """
    if baseline is None:
        baseline = max(s["remaining"] for s in scenarios)  # status quo assumed as max.
    labels = [s["scenario"] for s in scenarios]
    remaining = [s["remaining"] for s in scenarios]
    colors = ['green' if rem >= threshold_fraction * baseline else 'red' for rem in remaining]
    fig = px.bar(x=labels, y=remaining, labels={'x': 'Scenario', 'y': 'Remaining Production (MW)'},
                 title="Preventive Maintenance Simulation")
    fig.update_traces(marker_color=colors)
    fig.add_hline(y=threshold_fraction * baseline,
                  line_dash="dot", annotation_text=f"Threshold ({threshold_fraction*100:.0f}%)", 
                  annotation_position="top left")
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    return fig

# -------------------------------
# 4. Infrastructure Investment & Capacity Planning
# -------------------------------
def simulate_investment(current_nodes):
    """
    Simulate infrastructure investment scenarios.
    Scenarios: Status Quo, Add 1 Unit, Add 2 Units, Decommission 1, Decommission 2.
    Returns a list of scenario dicts.
    """
    scenarios = []
    baseline = current_nodes.copy()
    baseline_total = sum(baseline)
    scenarios.append({
        "scenario": "Status Quo",
        "nodes": baseline,
        "total": baseline_total,
        "change": 0
    })
    # Adding new unit (assumed capacity 200)
    new_unit = 200
    for add in [1, 2]:
        new_nodes = baseline.copy() + [new_unit] * add
        total = sum(new_nodes)
        scenarios.append({
            "scenario": f"Add {add} Unit{'s' if add > 1 else ''}",
            "nodes": new_nodes,
            "total": total,
            "change": total - baseline_total
        })
    # Decommission: remove smallest 1 or 2 units (if possible)
    if len(baseline) > 1:
        sorted_nodes = sorted(baseline)
        new_nodes = baseline.copy()
        new_nodes.remove(sorted_nodes[0])
        total = sum(new_nodes)
        scenarios.append({
            "scenario": "Decommission 1 Unit",
            "nodes": new_nodes,
            "total": total,
            "change": total - baseline_total
        })
    if len(baseline) > 2:
        sorted_nodes = sorted(baseline)
        new_nodes = baseline.copy()
        for n in sorted_nodes[:2]:
            new_nodes.remove(n)
        total = sum(new_nodes)
        scenarios.append({
            "scenario": "Decommission 2 Units",
            "nodes": new_nodes,
            "total": total,
            "change": total - baseline_total
        })
    return scenarios

def create_investment_bar_chart(scenarios):
    labels = [s["scenario"] for s in scenarios]
    totals = [s["total"] for s in scenarios]
    baseline = scenarios[0]["total"]
    fig = px.bar(x=labels, y=totals,
                 labels={'x': 'Scenario', 'y': 'Total Production Capacity (MW)'},
                 title="Infrastructure Investment Scenarios")
    fig.add_hline(y=baseline, line_dash="dash", 
                  annotation_text=f"Baseline: {baseline} MW", annotation_position="bottom right")
    # Annotate bars with total and change
    annotations = []
    for i, s in enumerate(scenarios):
        annotations.append(dict(x=labels[i], y=totals[i]+5, text=f"{totals[i]} MW ({s['change']:+})", showarrow=False))
    fig.update_layout(annotations=annotations)
    return fig

# -------------------------------
# 5 & 6. Cyber-Physical System Security & Cascading Failure Simulation
# -------------------------------
class CyberPhysicalGrid:
    def __init__(self):
        self.G = nx.Graph()
        self._initialize_grid()

    def _initialize_grid(self):
        """Initialize a sample grid topology."""
        self.G.clear()
        # Generators
        self.G.add_node("G1", type="generator", capacity=200)
        self.G.add_node("G2", type="generator", capacity=150)
        # Substations
        for i in range(1, 5):
            self.G.add_node(f"S{i}", type="substation", capacity=100)
        # Consumers
        for i in range(1, 4):
            self.G.add_node(f"C{i}", type="consumer", demand=80)
        # Edges (Transmission Lines)
        connections = [("G1", "S1"), ("G1", "S2"),
                       ("G2", "S3"), ("G2", "S4"),
                       ("S1", "S2"), ("S2", "S3"), ("S3", "S4"),
                       ("S1", "C1"), ("S3", "C2"), ("S4", "C3")]
        for u, v in connections:
            self.G.add_edge(u, v, capacity=100)

    def simulate_attack(self, failed_nodes=None, failed_edges=None):
        if failed_nodes:
            self.G.remove_nodes_from(failed_nodes)
        if failed_edges:
            self.G.remove_edges_from(failed_edges)

    def evaluate_system(self):
        remaining_gen = sum(data.get('capacity', 0) for n, data in self.G.nodes(data=True) if data.get('type') == 'generator')
        remaining_demand = sum(data.get('demand', 0) for n, data in self.G.nodes(data=True) if data.get('type') == 'consumer')
        components = list(nx.connected_components(self.G))
        isolated = 0
        for comp in components:
            # If there are consumers and no generators in a component, they are unserved.
            if any(self.G.nodes[n]['type'] == 'consumer' for n in comp) and not any(self.G.nodes[n]['type'] == 'generator' for n in comp):
                isolated += sum(self.G.nodes[n].get('demand', 0) for n in comp)
        return {"generation": remaining_gen, "demand": remaining_demand, "unserved": isolated}

    def get_network_figure(self, title="Grid Topology"):
        pos = nx.spring_layout(self.G, seed=42)
        node_x, node_y, node_text, node_color = [], [], [], []
        for node, data in self.G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            typ = data.get('type')
            if typ == 'generator':
                node_color.append("green")
                node_text.append(f"{node}<br>Gen: {data.get('capacity')}")
            elif typ == 'consumer':
                node_color.append("red")
                node_text.append(f"{node}<br>Demand: {data.get('demand')}")
            else:
                node_color.append("orange")
                node_text.append(f"{node}<br>Substation")
        edge_x, edge_y = [], []
        for u, v in self.G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=2, color='#888'),
                                hoverinfo='none', mode='lines')
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                text=node_text, hoverinfo='text',
                                marker=dict(color=node_color, size=30, line_width=2))
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title=dict(text=title, font=dict(size=16)),
                                         showlegend=False,
                                         hovermode='closest',
                                         margin=dict(b=20, l=5, r=5, t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        return fig

# For cascading failure simulation
class CascadingFailureSimulator(CyberPhysicalGrid):
    def __init__(self, failure_probability=0.3):
        super().__init__()
        self.failure_probability = failure_probability
        self.time_step = 0

    def reset_simulation(self):
        self._initialize_grid()
        self.time_step = 0

    def cascade_step(self):
        nodes_to_fail = []
        for n, data in list(self.G.nodes(data=True)):
            if data.get("type") in ["substation", "consumer"]:
                if random.random() < self.failure_probability:
                    nodes_to_fail.append(n)
        if nodes_to_fail:
            self.G.remove_nodes_from(nodes_to_fail)
            logging.info(f"Time {self.time_step}: Nodes failed: {nodes_to_fail}")
        else:
            logging.info(f"Time {self.time_step}: No failures this step.")
        self.time_step += 1

    def run_cascade(self, steps=1):
        for _ in range(steps):
            self.cascade_step()


######################################
# Dashboard Layout with Dash
######################################
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Multi-Use-Case Simulation Dashboard"

# Build multi-tab layout. Each tab represents one use case.
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

######################################
# Callbacks for Updating Each Tab
######################################

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
@app.callback(Output("n1-graph", "figure"),
              Input("btn-n1", "n_clicks"))
def update_n1(n_clicks):
    if not n_clicks:
        # Default dummy graph
        return go.Figure()
    # Example test data for production nodes
    test_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
    ids, lost_ratios = simulate_n1_contingency(test_nodes)
    fig = create_n1_bar_chart(ids, lost_ratios)
    return fig

# Callback for Renewable simulation
@app.callback(Output("renewable-graph", "figure"),
              Input("btn-renewable", "n_clicks"))
def update_renewable(n_clicks):
    if not n_clicks:
        return go.Figure()
    test_sources = {"Solar":100, "Wind":150, "Hydro":80}
    scenario_ids, total_prod, _ = simulate_renewable_sensitivity(test_sources, num_scenarios=50, seed=42)
    fig = create_renewable_scatter(scenario_ids, total_prod)
    return fig

# Callback for Maintenance simulation
@app.callback(Output("maintenance-graph", "figure"),
              Input("btn-maintenance", "n_clicks"))
def update_maintenance(n_clicks):
    if not n_clicks:
        return go.Figure()
    test_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
    scenarios = simulate_maintenance(test_nodes, max_outage=2)
    fig = create_maintenance_bar_chart(scenarios, threshold_fraction=0.80, baseline=sum(test_nodes))
    return fig

# Callback for Investment simulation
@app.callback(Output("investment-graph", "figure"),
              Input("btn-investment", "n_clicks"))
def update_investment(n_clicks):
    if not n_clicks:
        return go.Figure()
    test_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
    scenarios = simulate_investment(test_nodes)
    fig = create_investment_bar_chart(scenarios)
    return fig

# Callback for Cyber-Physical Security simulation
@app.callback(Output("cyber-graph", "figure"),
              Output("cyber-eval", "children"),
              Input("btn-cyber", "n_clicks"))
def update_cyber(n_clicks):
    if not n_clicks:
        return go.Figure(), ""
    grid = CyberPhysicalGrid()
    # Simulate attack: remove one generator and one substation, and one edge.
    grid.simulate_attack(failed_nodes=["G1", "S2"], failed_edges=[("S2", "S3")])
    fig = grid.get_network_figure("Grid After Cyber Attack")
    eval_data = grid.evaluate_system()
    eval_text = (
        f"Generation: {eval_data['generation']} MW, "
        f"Demand: {eval_data['demand']} MW, "
        f"Unserved Demand: {eval_data['unserved']} MW"
    )
    return fig, eval_text

# Single callback for Cascading Failure Simulation to avoid multiple outputs conflict.
@app.callback(
    Output("cascade-graph", "figure"),
    Output("cascade-eval", "children"),
    Input("btn-reset-cascade", "n_clicks"),
    Input("btn-cascade-step", "n_clicks"),
    Input("btn-cascade-multi", "n_clicks"),
    State("input-prob", "value"),
    State("input-steps", "value"),
    prevent_initial_call=True
)
def cascade_actions(n_reset, n_step, n_multi, prob_value, steps):
    ctx = callback_context
    if not ctx.triggered:
        return go.Figure(), ""
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "btn-reset-cascade":
        cascade_simulator.failure_probability = float(prob_value)
        cascade_simulator.reset_simulation()
    elif button_id == "btn-cascade-step":
        cascade_simulator.cascade_step()
    elif button_id == "btn-cascade-multi":
        cascade_simulator.run_cascade(steps=int(steps))
    
    fig = cascade_simulator.get_network_figure(f"Cascading Failure Simulation (Step {cascade_simulator.time_step})")
    eval_data = cascade_simulator.evaluate_system()
    eval_text = f"Time: {cascade_simulator.time_step} | Gen: {eval_data['generation']} MW, Unserved: {eval_data['unserved']} MW"
    return fig, eval_text

# Global instance for cascading simulation
cascade_simulator = CascadingFailureSimulator(failure_probability=0.3)

######################################
# Run the Dashboard
######################################
if __name__ == "__main__":
    app.run(debug=True)
