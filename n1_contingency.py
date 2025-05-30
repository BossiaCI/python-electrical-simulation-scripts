"""
N-1 Contingency Analysis with Decision Support and Outcome Visualization

This script simulates the removal of each production node (N-1 analysis) in an electrical grid,
supports decision-making by flagging critical nodes based on a configurable threshold, and produces
visualizations for both the simulation and the decision support outcome.

Python script that implements an N‑1 contingency analysis for an electrical grid and then supports decision-making by ranking and 
flagging nodes based on their impact on overall production. The script is divided into three main parts:

Simulation: It computes the loss in production when each node is removed.

Decision Support: It analyzes the simulation output to determine which nodes are critical (i.e. whose loss exceeds a configurable threshold).

Outcome Visualization: It generates plots that show the simulation results and highlights the nodes requiring action.

Usage:
    python n1_contingency.py
"""

import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

class GridSimulator:
    """
    Simulator for an electrical grid's production nodes.

    Attributes:
        nodes (np.ndarray): Array of production capacities for each node.
    """
    def __init__(self, nodes):
        """
        Initialize the simulator.

        Args:
            nodes (list or np.ndarray): Production capacities for grid nodes.
        """
        if not nodes:
            raise ValueError("Nodes list cannot be empty.")
        self.nodes = np.array(nodes, dtype=float)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized GridSimulator with %d nodes.", self.nodes.size)

    def run_contingency_analysis(self):
        """
        Simulate the removal of each node to compute its impact on total production.

        Returns:
            dict: A dictionary with:
                - 'node_indices': list of node indices (0-indexed)
                - 'lost_production_ratio': list of lost production percentages when each node fails
                - 'absolute_loss': list of absolute production lost when each node fails.
        """
        total_production = self.nodes.sum()
        self.logger.info("Total grid production: %.2f", total_production)

        if total_production == 0:
            raise ValueError("Total production cannot be zero.")

        node_indices = []
        lost_production_ratio = []
        absolute_loss = []

        # Loop over each node and simulate its removal
        for idx in range(self.nodes.size):
            lost = self.nodes[idx]
            loss_ratio = (lost / total_production) * 100
            node_indices.append(idx)
            lost_production_ratio.append(loss_ratio)
            absolute_loss.append(lost)

            self.logger.debug("Node %d: Production = %.2f, Loss Ratio = %.2f%%", idx, lost, loss_ratio)

        self.logger.info("Completed N-1 contingency analysis.")
        return {
            "node_indices": node_indices,
            "lost_production_ratio": lost_production_ratio,
            "absolute_loss": absolute_loss
        }

def decision_support(simulation_results, threshold_ratio=20.0):
    """
    Analyzes simulation results to flag critical nodes whose loss production ratio exceeds a threshold.

    Args:
        simulation_results (dict): Dictionary returned by run_contingency_analysis.
        threshold_ratio (float): The loss percentage threshold to consider a node as critical.

    Returns:
        dict: A dictionary with keys:
            - 'critical_nodes': list of node indices flagged as critical.
            - 'all_results': simulation_results (for reference).
    """
    critical_nodes = []
    for idx, loss_ratio in zip(simulation_results["node_indices"], simulation_results["lost_production_ratio"]):
        if loss_ratio >= threshold_ratio:
            critical_nodes.append(idx)
            logging.info("Node %d flagged as critical (loss ratio: %.2f%%)", idx, loss_ratio)
    return {"critical_nodes": critical_nodes, "all_results": simulation_results}

def plot_simulation_results(results, title="N-1 Contingency Analysis Simulation"):
    """
    Plots the simulation results as a bar graph representing the lost production ratio per node.

    Args:
        results (dict): Dictionary with simulation results.
        title (str): Title of the plot.
    """
    node_indices = results["node_indices"]
    lost_ratios = results["lost_production_ratio"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(node_indices, lost_ratios, color='cornflowerblue', edgecolor='black')
    plt.xlabel("Node Index (Failure Simulation)")
    plt.ylabel("Lost Production Ratio (%)")
    plt.title(title)
    plt.ylim(0, max(lost_ratios) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate each bar with its loss ratio percentage
    for bar, ratio in zip(bars, lost_ratios):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{ratio:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_decision_support_outcome(decision_data, title="Decision Support Outcome"):
    """
    Generates a pie chart showing the proportion of critical nodes compared to the rest.
    
    Args:
        decision_data (dict): Dictionary from decision_support containing 'critical_nodes' and full results.
        title (str): Title of the chart.
    """
    all_nodes = decision_data["all_results"]["node_indices"]
    critical_nodes = decision_data["critical_nodes"]
    non_critical = list(set(all_nodes) - set(critical_nodes))
    
    labels = ['Critical Nodes', 'Non-critical Nodes']
    sizes = [len(critical_nodes), len(non_critical)]
    colors = ['salmon', 'lightgreen']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(title)
    plt.axis('equal')
    plt.show()

def main():
    try:
        # Example input data: production capacities (e.g. in MW) of different nodes in the grid
        production_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
        simulator = GridSimulator(production_nodes)

        # 1. Simulation: Run N-1 contingency analysis
        simulation_results = simulator.run_contingency_analysis()
        plot_simulation_results(simulation_results)

        # 2. Decision Support: Analyze simulation results to flag critical nodes
        # Here, we define a threshold that flags any node contributing 20% or more of total production as critical.
        decision_data = decision_support(simulation_results, threshold_ratio=20.0)

        # 3. Outcome Visualization: Plot decision support results
        plot_decision_support_outcome(decision_data)

        # Outcome: Log results to provide decision context
        critical_nodes = decision_data["critical_nodes"]
        if critical_nodes:
            logging.info("The following nodes are critical: %s", critical_nodes)
            logging.info("Recommended action: Consider reinforcing or adding redundancy to critical nodes.")
        else:
            logging.info("No critical nodes identified; current grid configuration is robust.")

    except Exception as error:
        logging.error("An error occurred during the N-1 contingency analysis: %s", error)
        sys.exit(1)

if __name__ == "__main__":
    main()
