# src/simulation_dashboard/domain/models.py
from itertools import combinations
import random
import networkx as nx
import numpy as np

class Grid:
    def __init__(self):
        self.graph = nx.Graph()
        self._initialize_grid()

    def _initialize_grid(self):
        """Initialize grid topology with generators, substations, and consumers."""
        self.graph.clear()
        # Add generators
        self.graph.add_node("G1", type="generator", capacity=200)
        self.graph.add_node("G2", type="generator", capacity=150)
        # Add substations
        for i in range(1, 5):
            self.graph.add_node(f"S{i}", type="substation", capacity=100)
        # Add consumers
        for i in range(1, 4):
            self.graph.add_node(f"C{i}", type="consumer", demand=80)
        # Create transmission lines
        connections = [
            ("G1", "S1"), ("G1", "S2"),
            ("G2", "S3"), ("G2", "S4"),
            ("S1", "S2"), ("S2", "S3"), ("S3", "S4"),
            ("S1", "C1"), ("S3", "C2"), ("S4", "C3")
        ]
        for u, v in connections:
            self.graph.add_edge(u, v, capacity=100)

    def simulate_attack(self, failed_nodes=None, failed_edges=None):
        """Simulate removal of nodes and/or edges."""
        if failed_nodes:
            self.graph.remove_nodes_from(failed_nodes)
        if failed_edges:
            self.graph.remove_edges_from(failed_edges)

    def evaluate(self):
        """Evaluate grid performance (remaining generation and unserved demand)."""
        remaining_gen = sum(
            data.get('capacity', 0)
            for n, data in self.graph.nodes(data=True)
            if data.get('type') == 'generator'
        )
        remaining_demand = sum(
            data.get('demand', 0)
            for n, data in self.graph.nodes(data=True)
            if data.get('type') == 'consumer'
        )
        components = list(nx.connected_components(self.graph))
        unserved = 0
        for comp in components:
            if any(self.graph.nodes[n]['type'] == 'consumer' for n in comp) and not any(self.graph.nodes[n]['type'] == 'generator' for n in comp):
                unserved += sum(self.graph.nodes[n].get('demand', 0) for n in comp)
        return {"generation": remaining_gen, "demand": remaining_demand, "unserved": unserved}


class CascadingFailureSimulator(Grid):
    def __init__(self, failure_probability=0.3):
        super().__init__()
        self.failure_probability = failure_probability
        self.time_step = 0

    def reset(self):
        self._initialize_grid()
        self.time_step = 0

    def cascade_step(self):
        """Perform one time step of cascading failure."""
        nodes_to_fail = []
        for node, data in list(self.graph.nodes(data=True)):
            if data.get("type") in ["substation", "consumer"]:
                if random.random() < self.failure_probability:
                    nodes_to_fail.append(node)
        if nodes_to_fail:
            self.graph.remove_nodes_from(nodes_to_fail)
        self.time_step += 1

    def run_cascade(self, steps=1):
        for _ in range(steps):
            self.cascade_step()


# ------------------------------
# N-1 Contingency Analysis Model
# ------------------------------
class N1Contingency:
    def __init__(self, production_nodes):
        """
        Args:
            production_nodes (list of float): Production capacities of each node.
        """
        if not production_nodes:
            raise ValueError("Production nodes list cannot be empty.")
        self.nodes = np.array(production_nodes, dtype=float)
        self.total = self.nodes.sum()
    
    def run(self):
        """
        Run N-1 contingency by removing each node once.
        
        Returns:
            dict: {
                "node_indices": list of node indices,
                "lost_ratios": list of lost production ratios (in %)
            }
        """
        node_indices = list(range(len(self.nodes)))
        lost_ratios = [(node / self.total) * 100 for node in self.nodes]
        return {"node_indices": node_indices, "lost_ratios": lost_ratios}


# ------------------------------
# Renewable Sensitivity Model
# ------------------------------
class RenewableSensitivity:
    def __init__(self, sources, num_scenarios=50, seed=None):
        """
        Args:
            sources (dict): Dictionary of renewable sources with their maximum capacities.
                            e.g. {"Solar": 100, "Wind": 150, "Hydro": 80}
            num_scenarios (int): Number of simulation scenarios.
            seed (int, optional): Seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        self.sources = sources
        self.num_scenarios = num_scenarios

    def run(self):
        """
        Perform sensitivity analysis by varying the weather multiplier for each source.
        
        Returns:
            dict: {
                "scenario_ids": list of scenario indices,
                "total_production": list of total production for each scenario,
                "multipliers": list of dictionaries (per scenario) mapping source to multiplier
            }
        """
        scenario_ids = list(range(self.num_scenarios))
        total_production = []
        multipliers = []
        for _ in scenario_ids:
            scenario_factors = {}
            prod = 0.0
            for src, cap in self.sources.items():
                # Weather multiplier between 0.2 (poor) and 1.0 (optimal)
                factor = np.random.Generator(0.2, 1.0)
                scenario_factors[src] = factor
                prod += cap * factor
            total_production.append(prod)
            multipliers.append(scenario_factors)
        return {
            "scenario_ids": scenario_ids,
            "total_production": total_production,
            "multipliers": multipliers
        }


# ------------------------------
# Preventive Maintenance Model
# ------------------------------
class MaintenanceSimulation:
    def __init__(self, production_nodes, max_outage=2):
        """
        Args:
            production_nodes (list of float): Production capacities for each grid node.
            max_outage (int): Maximum number of nodes to be offline simultaneously.
        """
        if not production_nodes:
            raise ValueError("Production nodes list cannot be empty.")
        self.nodes = production_nodes
        self.total = sum(self.nodes)
        self.max_outage = max_outage

    def run(self):
        """
        Simulate maintenance scenarios by taking combinations of nodes offline.
        
        Returns:
            list of dict: Each scenario dictionary contains:
                - "scenario": String representation of failed node indices,
                - "offline": Total production lost,
                - "remaining": Remaining production after outage,
                - "loss_ratio": Loss as a percentage of total production.
        """
        scenarios = []
        for outage_count in range(1, self.max_outage + 1):
            for off in combinations(range(len(self.nodes)), outage_count):
                offline = sum([self.nodes[i] for i in off])
                remaining = self.total - offline
                loss_ratio = (offline / self.total) * 100
                scenarios.append({
                    "scenario": str(off),
                    "offline": offline,
                    "remaining": remaining,
                    "loss_ratio": loss_ratio
                })
        return scenarios


# ------------------------------
# Infrastructure Investment Model
# ------------------------------
class InvestmentScenario:
    def __init__(self, current_nodes):
        """
        Args:
            current_nodes (list of float): Current production capacities of grid nodes.
        """
        if not current_nodes:
            raise ValueError("Current nodes list cannot be empty.")
        self.current_nodes = current_nodes
        self.baseline_total = sum(current_nodes)

    def run(self):
        """
        Simulate investment scenarios such as status quo, adding new units, and decommissioning units.
        
        Returns:
            list of dict: Each scenario includes:
                - "scenario": Description,
                - "nodes": List of production capacities after the change,
                - "total": Total production capacity,
                - "change": Change vs. baseline (MW)
        """
        scenarios = []
        baseline = self.current_nodes.copy()
        baseline_total = self.baseline_total
        
        # Status Quo
        scenarios.append({
            "scenario": "Status Quo",
            "nodes": baseline,
            "total": baseline_total,
            "change": 0
        })

        # Adding new production units; assume each new unit adds 200 MW.
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

        # Decommissioning: Remove one or two smallest capacity nodes.
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
            for x in sorted_nodes[:2]:
                new_nodes.remove(x)
            total = sum(new_nodes)
            scenarios.append({
                "scenario": "Decommission 2 Units",
                "nodes": new_nodes,
                "total": total,
                "change": total - baseline_total
            })
        return scenarios

