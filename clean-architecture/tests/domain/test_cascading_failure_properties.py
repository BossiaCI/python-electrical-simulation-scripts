from hypothesis import given, settings
import hypothesis.strategies as st
from src.domain.models import CascadingFailureSimulation

@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=30),
    st.integers(min_value=1, max_value=20)
)
@settings(deadline=None)
def test_cascading_failure_timeline_and_survivability(failure_probs, time_steps):
    sim = CascadingFailureSimulation(failure_probs, time_steps)
    result = sim.run()
    assert "timeline" in result
    assert len(result["timeline"]) == time_steps
    assert all(isinstance(step, dict) for step in result["timeline"])
