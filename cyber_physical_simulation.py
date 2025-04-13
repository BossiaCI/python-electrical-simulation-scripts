"""
Here’s a production-ready Python simulation to model cyber-physical stress testing in power systems. 
This script simulates a grid with multiple generation and transmission nodes and introduces both physical failures 
and cyber-induced coordinated attacks. It assesses the grid's resilience based on failure propagation and system capacity loss.

Features:
Models grid topology using a graph.

Simulates random node/edge failures (physical or cyber).

Evaluates system capacity after failures.

Visualizes the grid before and after attack.

Includes test data for running scenarios.


Testing Data Embedded
The grid includes:

2 generators (G1, G2)

4 substations (S1–S4)

3 consumers (C1–C3)

Realistic transmission layout

Failure of nodes/edges (e.g., cyber-attack simulation on G1, S2, line between S2 and S3)




Usage:
    python cyber_physical_simulation.py
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import logging

logging.basicConfig(level=logging.INFO)

class CyberPhysicalGrid:
    def __init__(self):
        self.G = nx.Graph()
        self._initialize_grid()

    def _initialize_grid(self):
        """
        Create a sample grid topology with capacities.
        Nodes: Generators (G), Substations (S), Consumers (C)
        Edges: Transmission lines with capacity limits.
        """
        # Generators
        self.G.add_node("G1", type="generator", capacity=200)
        self.G.add_node("G2", type="generator", capacity=150)

        # Substations
        for i in range(1, 5):
            self.G.add_node(f"S{i}", type="substation", capacity=100)

        # Consumers
        for i in range(1, 4):
            self.G.add_node(f"C{i}", type="consumer", demand=80)

        # Transmission Lines
        connections = [
            ("G1", "S1"), ("G1", "S2"),
            ("G2", "S3"), ("G2", "S4"),
            ("S1", "S2"), ("S2", "S3"), ("S3", "S4"),
            ("S1", "C1"), ("S3", "C2"), ("S4", "C3")
        ]
        for u, v in connections:
            self.G.add_edge(u, v, capacity=100)

    def simulate_attack(self, failed_nodes=None, failed_edges=None):
        """
        Simulate a cyber or physical attack by removing nodes or edges.

        Args:
            failed_nodes (list): List of node IDs to simulate failure.
            failed_edges (list): List of edge tuples to simulate line failure.
        """
        if failed_nodes:
            self.G.remove_nodes_from(failed_nodes)
            logging.info(f"Simulated failure on nodes: {failed_nodes}")
        if failed_edges:
            self.G.remove_edges_from(failed_edges)
            logging.info(f"Simulated failure on transmission lines: {failed_edges}")

    def evaluate_system(self):
        """
        Evaluate the remaining generation vs. demand after failure.

        Returns:
            dict: Total generation, demand, and loss due to disconnection.
        """
        remaining_gen = sum(data['capacity'] for n, data in self.G.nodes(data=True) if data['type'] == 'generator')
        remaining_demand = sum(data['demand'] for n, data in self.G.nodes(data=True) if data['type'] == 'consumer')
        connected_components = list(nx.connected_components(self.G))

        isolated_demand = 0
        for component in connected_components:
            if any(self.G.nodes[n]['type'] == 'consumer' for n in component) and \
               not any(self.G.nodes[n]['type'] == 'generator' for n in component):
                # Consumers without a generator in their subgraph
                isolated_demand += sum(
                    self.G.nodes[n].get('demand', 0) for n in component
                )

        logging.info(f"Total Gen: {remaining_gen} MW, Total Demand: {remaining_demand} MW")
        logging.info(f"Isolated demand (unserved): {isolated_demand} MW")
        return {
            "total_generation": remaining_gen,
            "total_demand": remaining_demand,
            "unserved_demand": isolated_demand
        }

    def plot_grid(self, title="Grid Topology"):
        """
        Visualize the grid using NetworkX layout.
        """
        pos = nx.spring_layout(self.G, seed=42)
        colors = []
        for n in self.G.nodes(data=True):
            if n[1]['type'] == 'generator':
                colors.append("green")
            elif n[1]['type'] == 'consumer':
                colors.append("red")
            else:
                colors.append("orange")

        plt.figure(figsize=(10, 6))
        nx.draw(self.G, pos, with_labels=True, node_color=colors, node_size=800, edge_color='gray')
        plt.title(title)
        plt.grid(True)
        plt.show()


def run_test():
    grid = CyberPhysicalGrid()

    # Plot before attack
    grid.plot_grid("Original Grid Topology")

    # Simulate a coordinated cyber attack:
    # Knock out one generator and a key substation and a line
    failed_nodes = ["G1", "S2"]
    failed_edges = [("S2", "S3")]

    grid.simulate_attack(failed_nodes, failed_edges)

    # Evaluate after failure
    result = grid.evaluate_system()

    print("\n--- Attack Impact Summary ---")
    print(f"Remaining Generation Capacity: {result['total_generation']} MW")
    print(f"Total Demand: {result['total_demand']} MW")
    print(f"Unserved Demand (due to disconnection): {result['unserved_demand']} MW")

    # Plot after attack
    grid.plot_grid("Grid Topology After Attack")


if __name__ == "__main__":
    run_test()
