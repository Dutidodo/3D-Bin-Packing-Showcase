# Kernel benchmark

- Date: 2026-11-11
- Machine: Windows-11-10.0.26300-SP0; Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- Python 3.12.7, NumPy 2.2.4, Numba 0.65.1
- Command: `python benchmark.py --end-to-end`
- First compiled call (compile or cache load): 1.05 s

Same function source, interpreted by CPython vs compiled by Numba (single-threaded, no fastmath), identical inputs and identical results. Query: a 3x4x5 box with all 6 orientations in a 40x40x40 container with 50% stability.

| boxes in container | candidate points | point x orientation pairs | Python per call | Numba per call | ratio |
|---:|---:|---:|---:|---:|---:|
| 25 | 67 | 402 | 25.52 ms | 40.7 µs | 627× |
| 100 | 272 | 1632 | 262.96 ms | 691.8 µs | 380× |
| 250 | 703 | 4218 | 1295.80 ms | 3448.5 µs | 376× |

Constructive pass on the demo instance (120 items, volume order): Python 0.89 s, Numba 0.041 s (22×), identical layout: True.
