"""
Benchmark: the same placement kernel, interpreted by CPython vs compiled by Numba.

    python benchmark.py                # kernel micro-benchmark on three container states
    python benchmark.py --end-to-end   # also one full constructive pass with each kernel

Both variants run the identical function source (binpack/kernel.py) on identical
NumPy inputs. The script checks that they return the same decision and writes
the measurements, with machine and library versions, to benchmarks/results.md.

Note: the interpreted baseline reads NumPy arrays element by element, which is
a slow path for CPython. The ratio measures this kernel in this setting; it is
not a general "Numba is N times faster than Python" claim.
"""
import argparse
import datetime
import os
import platform
import random
import statistics
import sys
import time

import numpy as np

from binpack.kernel import NUMBA_AVAILABLE, best_position, get_kernel, warmup
from binpack.model import ContainerType, Instance, Item, make_synthetic_instance
from binpack.packer import Bin, Packer


def per_call(fn, args, min_time: float) -> float:
    """Seconds per call, measured over at least `min_time` seconds."""
    reps, t0 = 0, time.perf_counter()
    while True:
        fn(*args)
        reps += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= min_time:
            return elapsed / reps


def dense_states(targets, seed: int):
    """Fill one large container with random boxes; snapshot it when it holds `targets` boxes."""
    rng = random.Random(seed)
    ctype = ContainerType("bench", width=40, depth=40, height=40, max_weight=1e9, cost=1, gravity=50)
    items = [Item(f"B{i:04d}", float(rng.randint(2, 7)), float(rng.randint(2, 7)),
                  float(rng.randint(2, 7)), 1.0, 1.0, (0, 1, 2, 3, 4, 5)) for i in range(3000)]
    packer = Packer(Instance("bench", [ctype], items), use_numba=NUMBA_AVAILABLE)
    b, snaps = Bin(ctype), {}
    for it in items:
        packer.try_insert(b, it)
        if b.n in targets and b.n not in snaps:
            snaps[b.n] = b.copy()
        if len(snaps) == len(targets):
            break
    return ctype, [snaps[n] for n in sorted(snaps)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5, help="timing repeats (median is reported)")
    ap.add_argument("--min-time", type=float, default=0.5, help="seconds per timing repeat")
    ap.add_argument("--end-to-end", action="store_true", help="also time a full constructive pass")
    ap.add_argument("--out", default=os.path.join("benchmarks", "results.md"))
    args = ap.parse_args()

    if not NUMBA_AVAILABLE:
        print("Numba is not installed: nothing to compare (pip install numba).")
        return 1
    import numba

    compiled = get_kernel(True)
    t0 = time.perf_counter()
    warmup(compiled)
    first_call = time.perf_counter() - t0

    ctype, states = dense_states((25, 100, 250), args.seed)
    query = np.array([[3.0, 4.0, 5.0], [4.0, 3.0, 5.0], [3.0, 5.0, 4.0],
                      [5.0, 3.0, 4.0], [4.0, 5.0, 3.0], [5.0, 4.0, 3.0]])  # a 3x4x5 box, 6 orientations

    rows = []
    for b in states:
        call = (b.boxes, b.n, b.points, b.m, query, float(ctype.depth), float(ctype.width),
                float(ctype.height), ctype.gravity / 100.0)
        if best_position(*call) != compiled(*call):
            print("MISMATCH between the Python and the compiled kernel")
            return 1
        py = statistics.median(per_call(best_position, call, args.min_time) for _ in range(args.repeats))
        nb = statistics.median(per_call(compiled, call, args.min_time) for _ in range(args.repeats))
        rows.append((b.n, b.m, b.m * len(query), py, nb))

    print(f"Kernel: same source, CPython vs Numba {numba.__version__} (median of {args.repeats})")
    print(f"First compiled call (compile or cache load): {first_call:.2f} s")
    print(f"{'boxes':>6}{'points':>8}{'candidates':>12}{'Python':>14}{'Numba':>14}{'ratio':>9}")
    for n, m, cand, py, nb in rows:
        print(f"{n:>6}{m:>8}{cand:>12}{py * 1e3:>11.2f} ms{nb * 1e6:>11.1f} us{py / nb:>8.0f}x")

    e2e = None
    if args.end_to_end:
        inst = make_synthetic_instance(seed=7, n_items=120)
        order = sorted(inst.items, key=lambda it: (-it.volume, it.id))
        times, layouts = {}, {}
        for label, use_numba in (("Python", False), ("Numba", True)):
            packer = Packer(inst, use_numba=use_numba)
            packer.warmup()
            t0 = time.perf_counter()
            bins = packer.construct(order)
            times[label] = time.perf_counter() - t0
            layouts[label] = [[(it.id, o, x, y, z) for it, o, x, y, z in b.placements] for b in bins]
        same = layouts["Python"] == layouts["Numba"]
        e2e = (times["Python"], times["Numba"], same, len(layouts["Numba"]))
        print(f"\nConstructive pass, {inst.name} (120 items, volume order): "
              f"Python {e2e[0]:.2f} s, Numba {e2e[1]:.3f} s, ratio {e2e[0] / e2e[1]:.0f}x, "
              f"{e2e[3]} containers, identical layout: {same}")
        if not same:
            return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Kernel benchmark\n\n")
        f.write(f"- Date: {datetime.date.today().isoformat()}\n")
        f.write(f"- Machine: {platform.platform()}; {platform.processor() or platform.machine()}\n")
        f.write(f"- Python {platform.python_version()}, NumPy {np.__version__}, Numba {numba.__version__}\n")
        f.write(f"- Command: `python {' '.join(sys.argv)}`\n")
        f.write(f"- First compiled call (compile or cache load): {first_call:.2f} s\n\n")
        f.write("Same function source, interpreted by CPython vs compiled by Numba (single-threaded, "
                "no fastmath), identical inputs and identical results. Query: a 3x4x5 box with all "
                "6 orientations in a 40x40x40 container with 50% stability.\n\n")
        f.write("| boxes in container | candidate points | point x orientation pairs | Python per call | Numba per call | ratio |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for n, m, cand, py, nb in rows:
            f.write(f"| {n} | {m} | {cand} | {py * 1e3:.2f} ms | {nb * 1e6:.1f} µs | {py / nb:.0f}× |\n")
        if e2e:
            f.write(f"\nConstructive pass on the demo instance (120 items, volume order): "
                    f"Python {e2e[0]:.2f} s, Numba {e2e[1]:.3f} s ({e2e[0] / e2e[1]:.0f}×), "
                    f"identical layout: {e2e[2]}.\n")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
