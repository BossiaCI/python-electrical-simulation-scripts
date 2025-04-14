import pytest
from src.domain.models import RenewableSensitivity

def test_renewable_sensitivity_basic():
    sources = {"Solar": 100, "Wind": 150, "Hydro": 80}
    sim = RenewableSensitivity(sources, num_scenarios=5, seed=42)
    result = sim.run()

    assert len(result["scenario_ids"]) == 5
    assert len(result["total_production"]) == 5
    assert all(0.2 <= m <= 1.0 for r in result["multipliers"] for m in r.values())
    assert all(tp > 0 for tp in result["total_production"])
    assert all(set(r.keys()) == set(sources.keys()) for r in result["multipliers"])

def test_renewable_sensitivity_empty_sources():
    with pytest.raises(ValueError):
        RenewableSensitivity({}, num_scenarios=5).run()

def test_renewable_sensitivity_negative_capacity():
    sources = {"Solar": -100, "Wind": 150}
    sim = RenewableSensitivity(sources, num_scenarios=3, seed=0)
    result = sim.run()
    assert len(result["total_production"]) == 3
    assert any(p < 0 for p in result["total_production"])