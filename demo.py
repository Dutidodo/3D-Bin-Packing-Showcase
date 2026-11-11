"""
Clean-room demo: extreme-point packing of a synthetic heterogeneous instance.

    python demo.py                  # pack, self-check, save CSV + PNGs to assets/demo
    python demo.py --gif --html     # also the build-up animation and the interactive 3D page
    python demo.py --no-numba       # same decisions with the pure-Python kernel (much slower)

The run is deterministic for a given --seed (no time limit by default), so it
gives the same solution on any machine, with or without Numba.
"""
import argparse
import os
import sys
import time

from binpack import viz
from binpack.checker import check_solution
from binpack.kernel import NUMBA_AVAILABLE
from binpack.model import make_synthetic_instance, save_solution_csv
from binpack.packer import Packer


def lower_bound(instance) -> float:
    """Continuous bound: every unit of volume (or weight) costs at least the cheapest rate."""
    types = instance.container_types
    by_volume = sum(it.volume for it in instance.items) * min(t.cost / t.volume for t in types)
    by_weight = sum(it.weight for it in instance.items) * min(t.cost / t.max_weight for t in types)
    return max(by_volume, by_weight)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7, help="instance and search seed")
    ap.add_argument("--items", type=int, default=120, help="number of items")
    ap.add_argument("--restarts", type=int, default=6, help="randomised orderings after the 3 fixed ones")
    ap.add_argument("--time-limit", type=float, default=None, help="optional cap in seconds (breaks determinism)")
    ap.add_argument("--no-numba", action="store_true", help="use the pure-Python kernel")
    ap.add_argument("--out", default=os.path.join("assets", "demo"), help="output folder")
    ap.add_argument("--gif", action="store_true", help="also render the build-up GIF")
    ap.add_argument("--html", action="store_true", help="also write the interactive plotly page")
    args = ap.parse_args()

    inst = make_synthetic_instance(seed=args.seed, n_items=args.items)
    print(f"Instance {inst.name}: {len(inst.items)} items, "
          f"{len(inst.container_types)} container types")
    for t in inst.container_types:
        print(f"  {t.name:<8} {t.depth:g}x{t.width:g}x{t.height:g}  max weight {t.max_weight:g}  "
              f"cost {t.cost:g}  stability {t.gravity:g}%")

    use_numba = NUMBA_AVAILABLE and not args.no_numba
    if not args.no_numba and not NUMBA_AVAILABLE:
        print("Numba not installed: using the pure-Python kernel (same result, slower).")
    packer = Packer(inst, use_numba=use_numba)
    if use_numba:
        print(f"Kernel: Numba (compile / cache load {packer.warmup():.2f} s)")
    else:
        print("Kernel: pure Python")

    t0 = time.perf_counter()
    bins, history = packer.solve(restarts=args.restarts, seed=args.seed, time_limit=args.time_limit)
    elapsed = time.perf_counter() - t0

    print(f"\n{'ordering':<18}{'constructive':>14}{'after local search':>20}{'containers':>12}")
    for name, c0, c1, nb in history:
        print(f"{name:<18}{c0:>14g}{c1:>20g}{nb:>12}")

    sol = packer.to_solution(bins)
    cost = sol.cost(inst)
    lb = lower_bound(inst)
    used = {}
    for name in sol.containers.values():
        used[name] = used.get(name, 0) + 1
    print(f"\nBest cost {cost:g} with {len(bins)} containers "
          f"({', '.join(f'{v}x {k}' for k, v in used.items())}); "
          f"lower bound {lb:.1f}; search time {elapsed:.2f} s")

    errors = check_solution(inst, sol)
    if errors:
        print(f"Self-check FAILED: {len(errors)} violations")
        for e in errors[:10]:
            print("  -", e)
        return 1
    print(f"Self-check OK: {len(sol.placements)} items, {len(sol.containers)} containers, "
          f"0 violations (bounds, overlaps, weight, value, orientations, stability)")

    os.makedirs(args.out, exist_ok=True)
    save_solution_csv(sol, os.path.join(args.out, "solution.csv"))
    containers = viz.containers_from_solution(inst, sol)
    colors = viz.color_map(containers)
    viz.render_fleet(containers, os.path.join(args.out, "fleet.png"), colors,
                     suptitle=f"{inst.name} · cost {cost:g} · {len(containers)} containers")
    hero = max(containers, key=lambda c: (c.fill, -c.index))
    viz.render_container(hero, os.path.join(args.out, "container.png"), colors)
    written = ["solution.csv", "fleet.png", "container.png"]
    if args.gif:
        viz.render_build_gif(hero, os.path.join(args.out, "build.gif"), colors)
        written.append("build.gif")
    if args.html:
        if viz.render_html(containers, os.path.join(args.out, "packing.html"), colors,
                           title=f"{inst.name} · cost {cost:g}"):
            written.append("packing.html")
        else:
            print("plotly not installed: skipped packing.html")
    print(f"Wrote {', '.join(written)} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
