
from src.domain.models import RenewableSensitivity
from hypothesis import given
import hypothesis.strategies as st

@given(
    st.dictionaries(
        keys=st.sampled_from(["Solar", "Wind", "Hydro", "Geo"]),
        values=st.floats(min_value=0, max_value=500),
        min_size=1
    ),
    st.integers(min_value=1, max_value=10)
)
def test_renewable_total_production_is_non_negative(sources, scenarios):
    sim = RenewableSensitivity(sources, num_scenarios=scenarios, seed=42)
    result = sim.run()

    assert all(p >= 0 for p in result["total_production"])
    assert len(result["scenario_ids"]) == scenarios
