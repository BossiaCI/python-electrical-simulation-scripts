#!/usr/bin/env python3
"""
Cyber-Physical System Security Dashboard with Cascading Failure Simulation

This application provides a dashboard to simulate cyber-physical attacks on an
electrical grid. The simulation extends to time-based cascading failures where,
at each time step, nodes (e.g., substations or consumers) may fail (i.e., be removed)
based on a specified probability. The dashboard interface, built with Dash,
allows interactive control of the simulation and visualizes the grid topology.


a dashboard interface for a cyber‐physical grid security simulation. 
In this enhanced version, we extend the previous simulation to include time‐based cascading failures. In the simulation, 
failures (such as random removals of substations or consumers) cascade over discrete time steps based on a user‐controlled probability.
 The dashboard (built with Dash and Plotly) allows you to:

Initialize and reset the grid simulation.

Start a cascading failure simulation that evolves over time.

Visualize the grid topology interactively at each simulation step.

Usage:
    python cyber_physical_dashboard.py
"""

import random
import logging
import networkx as nx
import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State
import dash_bootstrap_components as dbc

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


# Define the base Cyber-Physical Grid simulation class
class CyberPhysicalGrid:
    def __init__(self):
        self.G = nx.Graph()
        self._initialize_grid()

    def _initialize_grid(self):
        """
        Create a sample grid topology with nodes and transmission lines.
        Node types:
          - "generator": Capacity in MW
          - "substation": A relay point (with a capacity attribute for uniformity)
          - "consumer": Demand in MW
        Edges represent transmission lines.
        """
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

        # Transmission Lines (edges)
        connections = [
            ("G1", "S1"), ("G1", "S2"),
            ("G2", "S3"), ("G2", "S4"),
            ("S1", "S2"), ("S2", "S3"), ("S3", "S4"),
            ("S1", "C1"), ("S3", "C2"), ("S4", "C3")
        ]
        for u, v in connections:
            self.G.add_edge(u, v, capacity=100)

    def simulate_attack(self, failed_nodes=None, failed_edges=None):
        """Simulate a (non-cascade) attack by removing nodes or edges."""
        if failed_nodes:
            self.G.remove_nodes_from(failed_nodes)
            logging.info(f"Simulated failure on nodes: {failed_nodes}")
        if failed_edges:
            self.G.remove_edges_from(failed_edges)
            logging.info(f"Simulated failure on edges: {failed_edges}")

    def evaluate_system(self):
        """Evaluate remaining generation, demand, and identify unserved demand."""
        remaining_gen = sum(
            data.get('capacity', 0) for n, data in self.G.nodes(data=True) if data.get('type') == 'generator'
        )
        remaining_demand = sum(
            data.get('demand', 0) for n, data in self.G.nodes(data=True) if data.get('type') == 'consumer'
        )
        connected_components = list(nx.connected_components(self.G))
        isolated_demand = 0
        for component in connected_components:
            if any(self.G.nodes[n]['type'] == 'consumer' for n in component) and \
               not any(self.G.nodes[n]['type'] == 'generator' for n in component):
                isolated_demand += sum(self.G.nodes[n].get('demand', 0) for n in component)
        logging.info(f"Evaluation: Generation={remaining_gen} MW, Demand={remaining_demand} MW, Unserved={isolated_demand} MW")
        return {
            "total_generation": remaining_gen,
            "total_demand": remaining_demand,
            "unserved_demand": isolated_demand
        }

    def get_network_figure(self, title="Grid Topology"):
        """Create a Plotly graph object representing the current grid."""
        pos = nx.spring_layout(self.G, seed=42)
        node_x, node_y, node_text, node_color = [], [], [], []
        # Build node positions and attributes
        for node, data in self.G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            n_type = data.get('type')
            if n_type == 'generator':
                node_color.append("green")
                node_text.append(f"{node}<br>Gen: {data.get('capacity')} MW")
            elif n_type == 'consumer':
                node_color.append("red")
                node_text.append(f"{node}<br>Demand: {data.get('demand')} MW")
            else:
                node_color.append("orange")
                node_text.append(f"{node}<br>Substation")

        # Build edge traces
        edge_x, edge_y = [], []
        for u, v in self.G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            hoverinfo="text",
            marker=dict(
                color=node_color,
                size=30,
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(text=title, font=dict(size=16)),
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        return fig


# Extend the grid simulation to include time-based cascading failures.
class CascadingFailureSimulator(CyberPhysicalGrid):
    def __init__(self, failure_probability=0.3):
        super().__init__()
        self.failure_probability = failure_probability
        self.time_step = 0

    def reset_simulation(self):
        """Reset the grid to its initial state and time step."""
        self._initialize_grid()
        self.time_step = 0
        logging.info("Simulation reset to initial grid state.")

    def cascade_step(self):
        """
        Perform one time step in the cascading failure simulation.
        For each node in the grid, if its type is either 'substation' or 'consumer',
        it fails with probability equal to failure_probability.
        """
        nodes_to_fail = []
        for node, data in list(self.G.nodes(data=True)):
            if data.get("type") in ["substation", "consumer"]:
                if random.random() < self.failure_probability:
                    nodes_to_fail.append(node)
        if nodes_to_fail:
            self.G.remove_nodes_from(nodes_to_fail)
            logging.info(f"Time {self.time_step}: Cascading failure removed nodes: {nodes_to_fail}")
        else:
            logging.info(f"Time {self.time_step}: No additional failures.")
        self.time_step += 1

    def run_cascade(self, steps=5):
        """
        Run the simulation for a given number of time steps.

        Args:
            steps (int): Number of cascade steps to perform.
        """
        for _ in range(steps):
            self.cascade_step()


# Build the Dash dashboard interface
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Cyber-Physical Grid Cascading Failure Dashboard"

# Create a global simulator instance with default failure probability.
simulator = CascadingFailureSimulator(failure_probability=0.3)

# Dashboard layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Cyber-Physical Grid Cascading Failure Simulation"), width=12)
    ], className="my-2"),
    dbc.Row([
        dbc.Col([
            dbc.Button("Reset Simulation", id="reset-btn", color="primary", className="me-2"),
            dbc.Button("Run Cascade Step", id="step-btn", color="danger", className="me-2"),
            dbc.InputGroup([
                dbc.InputGroupText("Failure Probability"),
                dbc.Input(id="probability-input", type="number", min=0, max=1, step=0.05, value=0.3)
            ], className="mb-2"),
            dbc.InputGroup([
                dbc.InputGroupText("Cascade Steps"),
                dbc.Input(id="steps-input", type="number", min=1, max=20, step=1, value=5)
            ], className="mb-2"),
            dbc.Button("Run Multiple Steps", id="run-steps-btn", color="warning", className="me-2")
        ], width=4),
        dbc.Col([
            dcc.Graph(id="grid-graph", figure=simulator.get_network_figure())
        ], width=8)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="evaluation-output"), width=12)
    ], className="my-2")
], fluid=True)


# Callback to reset the simulation.
@app.callback(
    Output("grid-graph", "figure"),
    Output("evaluation-output", "children"),
    Input("reset-btn", "n_clicks"),
    State("probability-input", "value"),
    prevent_initial_call=True
)
def reset_simulation(n_clicks, prob_value):
    simulator.failure_probability = float(prob_value)
    simulator.reset_simulation()
    fig = simulator.get_network_figure("Grid Topology (Reset)")
    eval_data = simulator.evaluate_system()
    eval_text = (
        f"Time Step: {simulator.time_step} | "
        f"Generation: {eval_data['total_generation']} MW | "
        f"Demand: {eval_data['total_demand']} MW | "
        f"Unserved Demand: {eval_data['unserved_demand']} MW"
    )
    return fig, eval_text


# Callback to run a single cascade step.
@app.callback(
    Output("grid-graph", "figure"),
    Output("evaluation-output", "children"),
    Input("step-btn", "n_clicks"),
    prevent_initial_call=True
)
def run_single_step(n_clicks):
    simulator.cascade_step()
    fig = simulator.get_network_figure(f"Grid Topology at Time Step {simulator.time_step}")
    eval_data = simulator.evaluate_system()
    eval_text = (
        f"Time Step: {simulator.time_step} | "
        f"Generation: {eval_data['total_generation']} MW | "
        f"Demand: {eval_data['total_demand']} MW | "
        f"Unserved Demand: {eval_data['unserved_demand']} MW"
    )
    return fig, eval_text


# Callback to run multiple cascade steps.
@app.callback(
    Output("grid-graph", "figure"),
    Output("evaluation-output", "children"),
    Input("run-steps-btn", "n_clicks"),
    State("steps-input", "value"),
    prevent_initial_call=True
)
def run_multiple_steps(n_clicks, steps):
    steps = int(steps)
    simulator.run_cascade(steps=steps)
    fig = simulator.get_network_figure(f"Grid Topology at Time Step {simulator.time_step}")
    eval_data = simulator.evaluate_system()
    eval_text = (
        f"Time Step: {simulator.time_step} | "
        f"Generation: {eval_data['total_generation']} MW | "
        f"Demand: {eval_data['total_demand']} MW | "
        f"Unserved Demand: {eval_data['unserved_demand']} MW"
    )
    return fig, eval_text


if __name__ == "__main__":
    app.run(debug=True)
