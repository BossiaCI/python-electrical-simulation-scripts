"""
Production-ready simulation and visualization module

This module simulates an N-1 contingency analysis for an electrical grid.
For each node (representing a production unit), the simulation removes the node (as if it has failed)
and computes the total loss in production relative to the full system operation.
It then visualizes the sensitivity of the grid production to each node’s failure.

Usage:
    python grid_simulation.py
"""

import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

# Configure logging for production-level diagnostic information
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
        Initialize the simulator with a set of nodes.
        
        Args:
            nodes (list or np.ndarray): Production capacities for grid nodes.
        """
        if not nodes:
            raise ValueError("The nodes list cannot be empty.")
        self.nodes = np.array(nodes, dtype=float)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized simulator with %d nodes.", self.nodes.size)
    
    def run_contingency_analysis(self):
        """
        Perform N-1 contingency analysis by simulating the failure of each node.
        
        Returns:
            results (dict): Dictionary with keys:
                'node_indices': list of node indices failed.
                'lost_production_ratio': list of production loss ratios (%) when each node fails.
        """
        total_production = self.nodes.sum()
        self.logger.info("Total production of the grid: %.2f", total_production)
        
        if total_production == 0:
            raise ValueError("Total production cannot be zero.")
        
        node_indices = []
        lost_production_ratio = []
        
        # Simulate the outage of each node individually
        for idx in range(self.nodes.size):
            production_without_node = total_production - self.nodes[idx]
            loss_ratio = (self.nodes[idx] / total_production) * 100
            node_indices.append(idx)
            lost_production_ratio.append(loss_ratio)
            
            self.logger.debug(
                "Node %d failure: node production = %.2f, lost ratio = %.2f%%",
                idx, self.nodes[idx], loss_ratio
            )
        
        results = {
            "node_indices": node_indices,
            "lost_production_ratio": lost_production_ratio
        }
        self.logger.info("Completed contingency analysis.")
        return results

def plot_contingency_results(results, title="N-1 Contingency Analysis"):
    """
    Generate a bar plot for the contingency analysis results.
    
    Args:
        results (dict): Dictionary with 'node_indices' and 'lost_production_ratio'.
        title (str): Title of the plot.
    """
    try:
        node_indices = results["node_indices"]
        lost_ratios = results["lost_production_ratio"]
    except KeyError as e:
        logging.error("Missing key in results dictionary: %s", e)
        raise
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(node_indices, lost_ratios, color='steelblue', edgecolor='black')
    plt.xlabel("Node Index (Failure Simulation)")
    plt.ylabel("Loss Ratio of Total Production (%)")
    plt.title(title)
    plt.ylim(0, max(lost_ratios) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with loss ratio percentage
    for bar, ratio in zip(bars, lost_ratios):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{ratio:.1f}%", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Example production capacities (in MW, for instance) of each node
        production_nodes = [120, 150, 80, 200, 50, 100]
        simulator = GridSimulator(production_nodes)
        simulation_results = simulator.run_contingency_analysis()
        plot_contingency_results(simulation_results)
    except Exception as error:
        logging.error("An error occurred during the simulation: %s", error)
        sys.exit(1)

if __name__ == "__main__":
    main()
