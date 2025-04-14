# from fastapi.testclient import TestClient
# from simulation_dashboard.api.app import app
# from hypothesis import given
# import hypothesis.strategies as st

# client = TestClient(app)

# @given(
#     st.lists(st.floats(min_value=-1e3, max_value=1e6), min_size=0, max_size=100)
# )
# def test_n1_api_fuzzing(nodes):
#     payload = {"nodes": nodes}
#     response = client.post("/api/simulation/n1", json=payload)
#     assert response.status_code in (200, 400)
# You can replicate this pattern for:

# /simulation/renewables

# /simulation/maintenance

# /simulation/investment