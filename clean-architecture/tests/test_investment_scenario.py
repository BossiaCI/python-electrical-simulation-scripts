import pytest
from src.domain.models import InvestmentScenario

def test_investment_scenarios():
    current_nodes = [100, 150, 200]
    sim = InvestmentScenario(current_nodes)
    scenarios = sim.run()

    scenario_names = [s["scenario"] for s in scenarios]
    assert "Status Quo" in scenario_names
    assert any("Add 1 Unit" in s for s in scenario_names)
    assert any("Add 2 Units" in s for s in scenario_names)
    assert any("Decommission 1 Unit" in s for s in scenario_names)
    assert any("Decommission 2 Units" in s for s in scenario_names)

    for scenario in scenarios:
        assert isinstance(scenario["nodes"], list)
        assert isinstance(scenario["total"], (int, float))
        assert isinstance(scenario["change"], (int, float))


def test_investment_empty_nodes():
    with pytest.raises(ValueError):
        InvestmentScenario([])

def test_investment_all_zero_nodes():
    sim = InvestmentScenario([0, 0, 0])
    result = sim.run()
    assert all(isinstance(r["total"], (int, float)) for r in result)
    assert all(r["total"] >= 0 for r in result)