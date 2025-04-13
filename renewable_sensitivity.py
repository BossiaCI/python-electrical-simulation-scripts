#!/usr/bin/env python3
"""
Renewable Energy Integration and Sensitivity Analysis Simulation

This script simulates the impact of weather variability on renewable energy sources
and analyzes its effect on grid stability and power quality. The simulation varies
the output of each renewable source using weather multipliers and calculates the total
production for a number of scenarios.

The simulation also provides decision support by flagging scenarios where the total
production falls below a specified threshold, and visualizes the results using scatter
plots.

Usage:
    python renewable_sensitivity.py
"""

import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

# Configure logging for production-level diagnostics.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

class RenewableGridSimulator:
    """
    Simulator for a renewable energy grid integrating multiple sources.

    Attributes:
        sources (dict): Dictionary of renewable sources with max production capacities (e.g., in MW).
    """

    def __init__(self, sources):
        """
        Initialize the simulator.

        Args:
            sources (dict): Dictionary with keys as source names and values as maximum capacities.
        """
        if not sources or not isinstance(sources, dict):
            raise ValueError("Sources must be provided as a non-empty dictionary.")
        self.sources = sources
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized RenewableGridSimulator with sources: %s", sources)

    def run_sensitivity_analysis(self, num_scenarios=50, seed=None):
        """
        Simulate a series of scenarios where the production of each source is modulated
        by a weather multiplier between 0.2 and 1.0.

        Args:
            num_scenarios (int): Number of simulation scenarios.
            seed (int): Optional random seed for reproducibility.

        Returns:
            dict: A dictionary with keys:
                - 'scenario_ids': list of scenario indices.
                - 'multipliers': list of dictionaries for each scenario (source -> multiplier).
                - 'total_production': list of total production values (sum of productions across sources).
        """
        if seed is not None:
            np.random.seed(seed)

        scenario_ids = list(range(num_scenarios))
        multipliers = []       # Each element is a dict: source -> multiplier
        total_production = []  # Total production for each scenario

        for scenario in scenario_ids:
            scenario_factors = {}
            production = 0.0
            # For each renewable source, simulate a weather multiplier.
            for source, capacity in self.sources.items():
                # Weather multiplier can vary between 0.2 (poor conditions) and 1.0 (optimal conditions)
                factor = np.random.uniform(0.2, 1.0)
                scenario_factors[source] = factor
                production += capacity * factor
            multipliers.append(scenario_factors)
            total_production.append(production)
            self.logger.debug("Scenario %d: multipliers = %s, total production = %.2f", scenario, scenario_factors, production)

        self.logger.info("Completed sensitivity analysis over %d scenarios.", num_scenarios)
        return {
            "scenario_ids": scenario_ids,
            "multipliers": multipliers,
            "total_production": total_production
        }

def decision_support(simulation_results, threshold_fraction=0.80):
    """
    Identify scenarios where total production is below a specified threshold.
    
    Args:
        simulation_results (dict): The dictionary returned by run_sensitivity_analysis.
        threshold_fraction (float): The fraction of maximum theoretical production to use as threshold.

    Returns:
        dict: Contains:
            - 'risky_scenarios': list of scenario indices where production falls below threshold.
            - 'threshold_value': The production threshold value.
            - 'all_results': The complete simulation results.
    """
    max_possible = sum(simulation_results["multipliers"][0][s] * cap
                       for s, cap in simulator.sources.items())  # if all multipliers were 1, max production
    # Alternatively, maximum theoretical production if all sources are at full capacity.
    max_theoretical = sum(simulator.sources.values())
    threshold_value = threshold_fraction * max_theoretical

    risky_scenarios = []
    for scenario, prod in zip(simulation_results["scenario_ids"], simulation_results["total_production"]):
        if prod < threshold_value:
            risky_scenarios.append(scenario)
            logging.info("Scenario %d flagged as at-risk (production: %.2f MW < threshold: %.2f MW)", scenario, prod, threshold_value)

    return {
        "risky_scenarios": risky_scenarios,
        "threshold_value": threshold_value,
        "all_results": simulation_results
    }

def plot_total_production(simulation_results, decision_data, title="Sensitivity Analysis: Total Production per Scenario"):
    """
    Plot the total production for each scenario, highlighting those below the threshold.

    Args:
        simulation_results (dict): Output from run_sensitivity_analysis.
        decision_data (dict): Output from decision_support.
        title (str): Title of the plot.
    """
    scenario_ids = simulation_results["scenario_ids"]
    total_prod = simulation_results["total_production"]
    risky = decision_data["risky_scenarios"]
    threshold = decision_data["threshold_value"]

    plt.figure(figsize=(10, 6))
    colors = ['red' if sid in risky else 'green' for sid in scenario_ids]
    plt.scatter(scenario_ids, total_prod, c=colors, s=80, edgecolors='k', alpha=0.7)
    plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=2, label=f"Threshold: {threshold:.2f} MW")
    plt.xlabel("Scenario ID")
    plt.ylabel("Total Production (MW)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_source_sensitivity(simulation_results, source_name, title_prefix="Sensitivity of Total Production to"):
    """
    For a chosen source, plot a scatter plot of its weather multiplier vs. total production.
    
    Args:
        simulation_results (dict): Output from run_sensitivity_analysis.
        source_name (str): The renewable source to analyze.
        title_prefix (str): Prefix for the plot title.
    """
    multipliers = [scenario[source_name] for scenario in simulation_results["multipliers"]]
    total_prod = simulation_results["total_production"]

    plt.figure(figsize=(10, 6))
    plt.scatter(multipliers, total_prod, color='blue', edgecolors='k', s=80, alpha=0.7)
    plt.xlabel(f"{source_name} Weather Multiplier")
    plt.ylabel("Total Grid Production (MW)")
    plt.title(f"{title_prefix} {source_name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def run_test_simulation():
    """
    Run the renewable energy sensitivity simulation with test data.
    """
    # Test data: renewable sources with maximum production capacities (in MW)
    test_sources = {
        "Solar": 100,   # e.g., 100 MW maximum
        "Wind": 150,    # e.g., 150 MW maximum
        "Hydro": 80     # e.g., 80 MW maximum
    }
    print("Test Renewable Sources:", test_sources)
    
    global simulator
    simulator = RenewableGridSimulator(test_sources)
    
    # Run sensitivity analysis over a series of scenarios.
    simulation_results = simulator.run_sensitivity_analysis(num_scenarios=50, seed=42)
    
    # Print a summary of simulation results.
    print("\nSimulation Results Summary:")
    print("Scenario IDs         :", simulation_results["scenario_ids"])
    print("Total Production (MW):", [round(prod, 2) for prod in simulation_results["total_production"]])
    
    # Decision Support: flag scenarios where total production is below 80% of maximum capacity.
    decision_data = decision_support(simulation_results, threshold_fraction=0.80)
    
    print("\nDecision Support:")
    if decision_data["risky_scenarios"]:
        print("At-risk Scenarios Identified:", decision_data["risky_scenarios"])
        print(f"Threshold Production Value: {decision_data['threshold_value']:.2f} MW")
    else:
        print("No at-risk scenarios identified; grid performance is robust.")
    
    # Visualization: Plot total production vs. scenario.
    plot_total_production(simulation_results, decision_data)
    
    # Visualization: Plot sensitivity for each renewable source.
    for source in test_sources.keys():
        plot_source_sensitivity(simulation_results, source_name=source)

if __name__ == "__main__":
    try:
        run_test_simulation()
    except Exception as error:
        logging.error("An error occurred during the simulation: %s", error)
        sys.exit(1)
