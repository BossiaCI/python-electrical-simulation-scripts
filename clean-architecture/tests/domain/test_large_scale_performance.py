import time
from src.domain.models import N1Contingency

def test_large_scale_n1_under_5s():
    nodes = [500.0] * 10000
    start = time.time()
    sim = N1Contingency(nodes)
    sim.run()
    assert time.time() - start < 5
