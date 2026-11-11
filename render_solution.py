"""
Check and draw solutions written in the course CSV format.

The instances and the competition solutions are NOT part of this repository;
point the script at your local copies.

    # one instance: independent feasibility check + fleet PNG, fullest container PNG
    python render_solution.py render <dataset_dir> <solution.csv> [--out assets/real]
                                     [--containers 0 1 2 ...] [--gif] [--html]

    # all instances: check every solution and print the cost table
    python render_solution.py summary <datasets_root> <results_dir> [--markdown]

<dataset_dir> contains items.csv and vehicles.csv; in `summary` the solution of
dataset NAME is the file matching sol_NAME_*.csv in <results_dir>.
"""
import argparse
import glob
import os
import sys

from binpack import viz
from binpack.checker import check_solution
from binpack.model import load_course_instance, load_course_solution


def cmd_render(args) -> int:
    inst = load_course_instance(args.dataset_dir)
    sol = load_course_solution(args.solution)
    errors = check_solution(inst, sol)
    print(f"{inst.name}: {len(inst.items)} items, {len(sol.containers)} containers, "
          f"cost {sol.cost(inst):,.2f}, violations {len(errors)}")
    for e in errors[:10]:
        print("  -", e)

    containers = viz.containers_from_solution(inst, sol)
    if args.containers:
        shown = [c for c in containers if c.index in set(args.containers)]
    else:
        shown = containers[:args.max_containers]
    colors = viz.color_map(containers)
    os.makedirs(args.out, exist_ok=True)
    name = inst.name
    # The drawings show the packing only: cost and container count stay in the console output.
    viz.render_fleet(shown, os.path.join(args.out, f"{name}_fleet.png"), colors,
                     suptitle=f"{name} · {len(shown)} containers shown")
    # the container drawn on its own: the one closest to its binding limit (volume or weight)
    hero = max(shown, key=lambda c: (c.fill, -c.index))
    print(f"Highlighted container: #{hero.index} ({hero.ctype.name}, "
          f"volume {hero.volume_fill:.0%}, weight {hero.weight_fill:.0%})")
    viz.render_container(hero, os.path.join(args.out, f"{name}_container.png"), colors)
    written = [f"{name}_fleet.png", f"{name}_container.png"]
    if args.gif:
        viz.render_build_gif(hero, os.path.join(args.out, f"{name}_build.gif"), colors)
        written.append(f"{name}_build.gif")
    if args.html:
        if viz.render_html(shown, os.path.join(args.out, f"{name}.html"), colors,
                           title=name):
            written.append(f"{name}.html")
        else:
            print("plotly not installed: skipped the HTML page")
    print(f"Wrote {', '.join(written)} to {args.out}")
    return 1 if errors else 0


def cmd_summary(args) -> int:
    rows, total, failed = [], 0.0, 0
    for ds in sorted(os.listdir(args.datasets_root)):
        ds_dir = os.path.join(args.datasets_root, ds)
        if not os.path.isfile(os.path.join(ds_dir, "items.csv")):
            continue
        matches = sorted(glob.glob(os.path.join(args.results_dir, f"sol_{ds}_*.csv")))
        if not matches:
            rows.append((ds, None, None, None, "no solution file"))
            failed += 1
            continue
        inst = load_course_instance(ds_dir)
        sol = load_course_solution(matches[0])
        errors = check_solution(inst, sol)
        cost = sol.cost(inst)
        total += cost
        failed += bool(errors)
        rows.append((ds, len(inst.items), len(sol.containers), cost,
                     "feasible" if not errors else f"{len(errors)} violations"))

    if args.markdown:
        print("| Instance | Items | Containers | Cost | Check |")
        print("|---|---:|---:|---:|---|")
        for ds, n, k, c, status in rows:
            print(f"| {ds} | {n if n is not None else '-'} | {k if k is not None else '-'} | "
                  f"{f'{c:,.2f}' if c is not None else '-'} | {status} |")
        print(f"| **Total** | | | **{total:,.2f}** | {len(rows) - failed}/{len(rows)} feasible |")
    else:
        print(f"{'instance':<12}{'items':>7}{'containers':>12}{'cost':>14}  check")
        for ds, n, k, c, status in rows:
            print(f"{ds:<12}{n if n is not None else '-':>7}{k if k is not None else '-':>12}"
                  f"{f'{c:,.2f}' if c is not None else '-':>14}  {status}")
        print(f"{'TOTAL':<31}{total:>14,.2f}  {len(rows) - failed}/{len(rows)} feasible")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="check and draw one solution")
    r.add_argument("dataset_dir")
    r.add_argument("solution")
    r.add_argument("--out", default=os.path.join("assets", "real"))
    r.add_argument("--containers", type=int, nargs="+", help="container indices to draw")
    r.add_argument("--max-containers", type=int, default=8,
                   help="how many containers to draw when --containers is not given")
    r.add_argument("--gif", action="store_true", help="build-up animation of the fullest container")
    r.add_argument("--html", action="store_true", help="interactive plotly page")
    r.set_defaults(func=cmd_render)

    s = sub.add_parser("summary", help="check all solutions and print the cost table")
    s.add_argument("datasets_root")
    s.add_argument("results_dir")
    s.add_argument("--markdown", action="store_true")
    s.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
