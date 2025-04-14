import pytest
from src.domain.models import MaintenanceSimulation

def test_maintenance_simulation_combinations():
    nodes = [100, 150, 200]
    sim = MaintenanceSimulation(nodes, max_outage=2)
    result = sim.run()

    # All combinations of 1 or 2 outages
    expected_scenarios = 3 + 3  # C(3,1) + C(3,2) = 6
    assert len(result) == expected_scenarios

    for r in result:
        assert "offline" in r and "remaining" in r and "loss_ratio" in r
        assert abs(r["remaining"] + r["offline"] - sum(nodes)) < 1e-5
        assert 0 <= r["loss_ratio"] <= 100


def test_maintenance_simulation_empty_nodes():
    with pytest.raises(ValueError):
        MaintenanceSimulation([], max_outage=1)

def test_maintenance_simulation_negative_nodes():
    nodes = [100, -100, 200]
    sim = MaintenanceSimulation(nodes, max_outage=1)
    result = sim.run()
    assert all("loss_ratio" in r for r in result)
    assert any(r["offline"] < 0 for r in result)
