#!/usr/bin/env python3
"""
Infrastructure Investment and Capacity Planning Simulation

This script simulates different long-term capacity planning strategies to assess the impact of
adding new production units or decommissioning old ones on the overall grid production capacity.
The simulation generates a visual report comparing the total production capacity under various
investment scenarios.

Usage:
    python infrastructure_investment.py
"""

import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

# Configure logging for production-level diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

class InfrastructureSimulator:
    """
    Simulator for infrastructure investment scenarios.

    Attributes:
        current_nodes (list): List of production capacities (MW) for current grid units.
    """

    def __init__(self, current_nodes):
        """
        Initialize the simulator with the current grid units.

        Args:
            current_nodes (list): Production capacities (MW) of current production units.
        """
        if not current_nodes:
            raise ValueError("The current_nodes list cannot be empty.")
        self.current_nodes = current_nodes
        self.baseline_total = sum(current_nodes)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized simulator with %d nodes; Baseline total production = %.2f MW",
                         len(current_nodes), self.baseline_total)

    def run_investment_scenarios(self):
        """
        Simulate various infrastructure scenarios.

        Scenarios:
            - "Status Quo": No change.
            - "Add 1 Unit": Add one new production unit.
            - "Add 2 Units": Add two new production units.
            - "Decommission 1 Unit": Remove the smallest production unit.
            - "Decommission 2 Units": Remove the two smallest production units.

        Returns:
            list of dicts: Each dict represents a scenario with:
                - 'scenario': Description
                - 'new_nodes': New list of production capacities after changes.
                - 'total_production': Sum of production capacities.
                - 'change': Difference versus baseline (MW).
        """
        scenarios = []
        baseline = self.current_nodes
        baseline_total = self.baseline_total

        # Scenario 1: Status Quo
        scenarios.append({
            "scenario": "Status Quo",
            "new_nodes": baseline.copy(),
            "total_production": baseline_total,
            "change": 0.0
        })

        # Scenario 2: Add 1 new production unit (assume a new unit capacity of 200 MW)
        new_unit_capacity = 200
        new_nodes = baseline.copy() + [new_unit_capacity]
        total_production = sum(new_nodes)
        scenarios.append({
            "scenario": "Add 1 Unit",
            "new_nodes": new_nodes,
            "total_production": total_production,
            "change": total_production - baseline_total
        })

        # Scenario 3: Add 2 new production units (each 200 MW)
        new_nodes = baseline.copy() + [new_unit_capacity, new_unit_capacity]
        total_production = sum(new_nodes)
        scenarios.append({
            "scenario": "Add 2 Units",
            "new_nodes": new_nodes,
            "total_production": total_production,
            "change": total_production - baseline_total
        })

        # Scenario 4: Decommission 1 unit (remove the smallest capacity unit)
        if len(baseline) > 1:
            sorted_nodes = sorted(baseline)
            node_to_remove = sorted_nodes[0]
            new_nodes = baseline.copy()
            new_nodes.remove(node_to_remove)
            total_production = sum(new_nodes)
            scenarios.append({
                "scenario": "Decommission 1 Unit",
                "new_nodes": new_nodes,
                "total_production": total_production,
                "change": total_production - baseline_total
            })

        # Scenario 5: Decommission 2 units (remove the two smallest capacity units)
        if len(baseline) > 2:
            sorted_nodes = sorted(baseline)
            nodes_to_remove = sorted_nodes[:2]
            new_nodes = baseline.copy()
            for node in nodes_to_remove:
                new_nodes.remove(node)
            total_production = sum(new_nodes)
            scenarios.append({
                "scenario": "Decommission 2 Units",
                "new_nodes": new_nodes,
                "total_production": total_production,
                "change": total_production - baseline_total
            })

        self.logger.info("Generated %d investment scenarios", len(scenarios))
        return scenarios

def plot_investment_scenarios(scenarios, title="Infrastructure Investment Scenarios"):
    """
    Plot the total production capacity for different investment scenarios.

    Args:
        scenarios (list of dicts): List of scenario dictionaries.
        title (str): Title of the plot.
    """
    scenario_names = [s["scenario"] for s in scenarios]
    production_values = [s["total_production"] for s in scenarios]
    baseline_total = scenarios[0]["total_production"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenario_names, production_values, color='skyblue', edgecolor='black')
    plt.axhline(y=baseline_total, color='gray', linestyle='--', linewidth=2, label=f"Baseline: {baseline_total} MW")
    plt.ylabel("Total Production Capacity (MW)")
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate each bar with the production value and change versus baseline.
    for bar, scenario in zip(bars, scenarios):
        height = bar.get_height()
        change = scenario["change"]
        annotation = f"{height:.0f} MW\n({change:+.0f} MW)"
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5, annotation,
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def run_test_simulation():
    """
    Run the infrastructure investment simulation with test data.
    """
    # Test Data: Current grid production capacities (in MW)
    test_nodes = [120, 150, 80, 200, 50, 100, 70, 180]
    print("Test Production Nodes:", test_nodes)

    simulator = InfrastructureSimulator(test_nodes)
    scenarios = simulator.run_investment_scenarios()

    # Print a summary of each scenario
    print("\nInvestment Scenarios:")
    for scenario in scenarios:
        print(f"Scenario: {scenario['scenario']}")
        print(f"  New Nodes: {scenario['new_nodes']}")
        print(f"  Total Production: {scenario['total_production']} MW")
        print(f"  Change vs Baseline: {scenario['change']:+.2f} MW")
        print("-" * 50)

    # Visualize the scenarios
    plot_investment_scenarios(scenarios)

if __name__ == "__main__":
    try:
        run_test_simulation()
    except Exception as error:
        logging.error("An error occurred during the simulation: %s", error)
        sys.exit(1)
