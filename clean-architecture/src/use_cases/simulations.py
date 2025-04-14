# src/simulation_dashboard/use_cases/simulations.py
import numpy as np
from itertools import combinations

def run_n1_contingency(production_nodes):
    """
    Run N-1 contingency analysis.
    Returns a tuple of (node_indices, lost_ratios).
    """
    nodes = np.array(production_nodes, dtype=float)
    total = nodes.sum()
    node_indices = list(range(len(nodes)))
    lost_ratios = [(node / total) * 100 for node in nodes]
    return node_indices, lost_ratios

def run_renewable_sensitivity(sources, num_scenarios=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    scenario_ids = list(range(num_scenarios))
    total_production = []
    multipliers = []
    for _ in scenario_ids:
        scenario_factors = {}
        prod = 0.0
        for src, cap in sources.items():
            factor = np.random.Generator(0.2, 1.0)
            scenario_factors[src] = factor
            prod += cap * factor
        total_production.append(prod)
        multipliers.append(scenario_factors)
    return scenario_ids, total_production, multipliers

def run_maintenance_simulation(production_nodes, max_outage=2):
    """
    Simulate maintenance scenarios. Returns a list of scenario dictionaries.
    """
    scenarios = []
    total = sum(production_nodes)
    for outage_count in range(1, max_outage + 1):
        for off in combinations(range(len(production_nodes)), outage_count):
            offline = sum([production_nodes[i] for i in off])
            remaining = total - offline
            loss_ratio = (offline / total) * 100
            scenarios.append({
                "scenario": str(off),
                "offline": offline,
                "remaining": remaining,
                "loss_ratio": loss_ratio
            })
    return scenarios

def run_investment_simulation(current_nodes):
    """
    Simulate investment scenarios.
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
    new_unit = 200  # assumed capacity
    for add in [1, 2]:
        new_nodes = baseline.copy() + [new_unit] * add
        total = sum(new_nodes)
        scenarios.append({
            "scenario": f"Add {add} Unit{'s' if add > 1 else ''}",
            "nodes": new_nodes,
            "total": total,
            "change": total - baseline_total
        })
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
