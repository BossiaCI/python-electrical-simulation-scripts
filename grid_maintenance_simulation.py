#!/usr/bin/env python3
"""
Grid Maintenance Simulation for Preventive Maintenance and Reliability Engineering

This script simulates the grid’s production performance under various maintenance scenarios
by taking one or more production units offline. It then assesses the risk by comparing the
remaining production against a minimum required threshold. Finally, it visualizes the results
to support maintenance planning and reduce the risk of unplanned outages.

Usage:
    python grid_maintenance_simulation.py
"""

import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Configure logging for production-level diagnostics.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

class GridMaintenanceSimulator:
    """
    Simulator for evaluating grid performance during maintenance scenarios.

    Attributes:
        nodes (np.ndarray): Array of production capacities for each node.
    """

    def __init__(self, nodes):
        """
        Initialize the simulator with production nodes.

        Args:
            nodes (list or np.ndarray): Production capacities (e.g., in MW) for grid nodes.
        """
        if not nodes:
            raise ValueError("Nodes list cannot be empty.")
        self.nodes = np.array(nodes, dtype=float)
        self.total_production = self.nodes.sum()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized with %d nodes, total production = %.2f MW", self.nodes.size, self.total_production)

    def run_maintenance_simulation(self, max_outage=2):
        """
        Simulate maintenance scenarios by taking various sets of nodes offline.

        Args:
            max_outage (int): Maximum number of nodes allowed to be offline simultaneously.

        Returns:
            list of dicts: Each entry represents a simulation scenario with:
                - 'scenario_id': A tuple of offline node indices.
                - 'offline_production': Total production lost due to offline nodes.
                - 'remaining_production': Production remaining after outage.
                - 'loss_ratio': Percentage loss of total production.
        """
        scenarios = []
        # Generate outages for 1 up to max_outage nodes
        for outage_count in range(1, max_outage + 1):
            for offline_nodes in combinations(range(self.nodes.size), outage_count):
                offline_prod = self.nodes[list(offline_nodes)].sum()
                remaining_prod = self.total_production - offline_prod
                loss_ratio = (offline_prod / self.total_production) * 100
                scenario = {
                    "scenario_id": offline_nodes,
                    "offline_production": offline_prod,
                    "remaining_production": remaining_prod,
                    "loss_ratio": loss_ratio
                }
                scenarios.append(scenario)
                self.logger.debug(
                    "Scenario %s: Offline Production = %.2f MW, Remaining = %.2f MW, Loss Ratio = %.2f%%",
                    offline_nodes, offline_prod, remaining_prod, loss_ratio
                )
        self.logger.info("Completed maintenance simulation for scenarios with up to %d nodes offline.", max_outage)
        return scenarios

def decision_support(scenarios, min_required_fraction=0.80):
    """
    Assess each maintenance scenario against the minimum required production threshold.

    Args:
        scenarios (list of dicts): List of simulation scenarios.
        min_required_fraction (float): Fraction of full production required (e.g., 0.80 for 80%).

    Returns:
        dict: With keys:
            - 'acceptable': List of scenario IDs that meet the production requirement.
            - 'risky': List of scenario IDs that fall below the production threshold.
            - 'threshold_value': The minimum required production value.
            - 'all_scenarios': Original list of scenarios.
    """
    # Full production when all nodes are online
    full_production = sum(scenario["remaining_production"] + scenario["offline_production"] for scenario in [scenarios[0]])  # same as simulator.total_production
    threshold_value = min_required_fraction * full_production

    acceptable = []
    risky = []
    for scenario in scenarios:
        if scenario["remaining_production"] >= threshold_value:
            acceptable.append(scenario["scenario_id"])
        else:
            risky.append(scenario["scenario_id"])
            logging.info("Scenario %s flagged as risky: Remaining Production = %.2f MW (< threshold: %.2f MW)",
                         scenario["scenario_id"], scenario["remaining_production"], threshold_value)
    return {
        "acceptable": acceptable,
        "risky": risky,
        "threshold_value": threshold_value,
        "all_scenarios": scenarios
    }

def plot_simulation_results(scenarios, threshold_value, title="Maintenance Simulation: Production Impact per Scenario"):
    """
    Plot the remaining production for each maintenance scenario as a bar chart.

    Args:
        scenarios (list of dicts): List of scenario dictionaries.
        threshold_value (float): The minimum required production value.
        title (str): Plot title.
    """
    # Create a label for each scenario (offline nodes as a string, e.g., "(1, 3)")
    labels = [str(scenario["scenario_id"]) for scenario in scenarios]
    remaining_production = [scenario["remaining_production"] for scenario in scenarios]

    # Determine bar colors: red if below threshold, green if acceptable.
    colors = ['red' if prod < threshold_value else 'green' for prod in remaining_production]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(scenarios)), remaining_production, color=colors, edgecolor='black')
    plt.axhline(y=threshold_value, color='gray', linestyle='--', linewidth=2, label=f"Min Required: {threshold_value:.2f} MW")
    plt.xlabel("Maintenance Scenario (Offline Nodes)")
    plt.ylabel("Remaining Production (MW)")
    plt.title(title)
    plt.xticks(range(len(scenarios)), labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Annotate each bar with remaining production value
    for bar, prod in zip(bars, remaining_production):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{prod:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def run_test_simulation():
    """
    Execute the maintenance simulation using test data.
    """
    # Test Data: Production capacities (in MW) of grid nodes (example values)
    test_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
    print("Test Production Nodes:", test_nodes)
    
    simulator = GridMaintenanceSimulator(test_nodes)
    
    # Run maintenance simulation for single unit outages (N-1) and selected dual outages (N-2)
    scenarios = simulator.run_maintenance_simulation(max_outage=2)
    
    # Print a summary of the simulation results
    print("\nSimulation Results:")
    for scenario in scenarios:
        print(f"Scenario {scenario['scenario_id']}: Offline = {scenario['offline_production']} MW, "
              f"Remaining = {scenario['remaining_production']} MW, Loss = {scenario['loss_ratio']:.1f}%")
    
    # Decision Support: Flag risky scenarios where remaining production falls below 80% of full production
    decision_data = decision_support(scenarios, min_required_fraction=0.80)
    print("\nDecision Support Summary:")
    if decision_data["risky"]:
        print("Risky Scenarios Identified:", decision_data["risky"])
        print(f"Production must remain above {decision_data['threshold_value']:.2f} MW.")
    else:
        print("All scenarios are acceptable; grid production remains robust.")
    
    # Visualization: Plot the remaining production per scenario with threshold indication.
    plot_simulation_results(scenarios, decision_data["threshold_value"])

if __name__ == "__main__":
    try:
        run_test_simulation()
    except Exception as error:
        logging.error("An error occurred during the simulation: %s", error)
        sys.exit(1)
