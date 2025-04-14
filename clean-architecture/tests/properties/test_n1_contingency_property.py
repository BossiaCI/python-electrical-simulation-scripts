from hypothesis import given
import hypothesis.strategies as st
from src.domain.models import N1Contingency

@given(st.lists(st.floats(min_value=0, max_value=1e6), min_size=1))
def test_n1_lost_ratios_sum_below_total(nodes):
    sim = N1Contingency(nodes)
    result = sim.run()

    total_capacity = result["total_capacity"]
    lost_ratios = result["lost_ratios"]

    assert all(0 <= ratio <= 100 for ratio in lost_ratios)
    assert total_capacity >= 0
