from hypothesis import given
import hypothesis.strategies as st
from src.domain.models import InvestmentScenario

@given(
    st.lists(st.floats(min_value=0, max_value=500), min_size=1, max_size=10)
)
def test_investment_scenarios_valid_total(nodes):
    sim = InvestmentScenario(nodes)
    result = sim.run()

    for scenario in result:
        assert scenario["total"] >= 0
        assert isinstance(scenario["nodes"], list)
