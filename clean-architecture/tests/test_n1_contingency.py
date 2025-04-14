import numpy as np
import pytest
from src.domain.models import N1Contingency
from src.use_cases import simulations

def test_n1_contingency():
    nodes = [100, 200, 300]
    indices, lost_ratios = simulations.run_n1_contingency(nodes)
    assert indices == [0, 1, 2]
    total = sum(nodes)
    expected = [(x / total)*100 for x in nodes]
    np.testing.assert_allclose(lost_ratios, expected)

    
def test_n1_contingency_empty_nodes():
    with pytest.raises(ValueError):
        N1Contingency([])

def test_n1_contingency_negative_values():
    nodes = [100, -50, 200]
    contingency = N1Contingency(nodes)
    result = contingency.run()
    assert all(isinstance(r, float) for r in result["lost_ratios"])
