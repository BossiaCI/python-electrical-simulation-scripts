from hypothesis import given
import hypothesis.strategies as st
from src.domain.models import MaintenanceSimulation

@given(
    st.lists(st.floats(min_value=0, max_value=1000), min_size=1, max_size=5),
    st.integers(min_value=1, max_value=3)
)
def test_maintenance_loss_ratio_bounds(nodes, max_outage):
    sim = MaintenanceSimulation(nodes, max_outage=min(max_outage, len(nodes)))
    result = sim.run()

    for r in result:
        assert 0 <= r["loss_ratio"] <= 100

