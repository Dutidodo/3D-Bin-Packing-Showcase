"""
Independent feasibility checker.

Plain Python, written separately from the packer and its kernel: it recomputes
every constraint from the solution alone, so a bug in the heuristic cannot hide
itself by being shared with the checker.
"""
from __future__ import annotations

from collections import defaultdict

from .model import Instance, Solution, oriented_extents


def check_solution(instance: Instance, solution: Solution, tol: float = 1e-6) -> list:
    """Return a list of human-readable violations (empty list = feasible)."""
    errors = []
    items = {it.id: it for it in instance.items}
    types = {t.name: t for t in instance.container_types}

    # containers: consecutive indices, known types
    if sorted(solution.containers) != list(range(len(solution.containers))):
        errors.append("container indices are not consecutive 0..n-1")
    for k, name in solution.containers.items():
        if name not in types:
            errors.append(f"container {k}: unknown type {name!r}")

    # items: each exactly once, allowed orientation, known container
    count = defaultdict(int)
    boxes = defaultdict(list)  # container -> [(item, x0, y0, z0, x1, y1, z1)]
    for p in solution.placements:
        count[p.item_id] += 1
        it = items.get(p.item_id)
        if it is None:
            errors.append(f"unknown item {p.item_id!r}")
            continue
        if p.orient not in it.rotations:
            errors.append(f"{it.id}: orientation {p.orient} not allowed {it.rotations}")
            continue
        if p.container not in solution.containers:
            errors.append(f"{it.id}: placed in undeclared container {p.container}")
            continue
        dx, dy, dz = oriented_extents(it.width, it.depth, it.height, p.orient)
        boxes[p.container].append((it, p.x, p.y, p.z, p.x + dx, p.y + dy, p.z + dz))
    for it in instance.items:
        if count[it.id] == 0:
            errors.append(f"{it.id}: not packed")
        elif count[it.id] > 1:
            errors.append(f"{it.id}: packed {count[it.id]} times")

    for k, content in boxes.items():
        t = types.get(solution.containers[k])
        if t is None:
            continue
        # capacity
        weight = sum(b[0].weight for b in content)
        value = sum(b[0].value for b in content)
        if weight > t.max_weight + tol:
            errors.append(f"container {k}: weight {weight:g} > {t.max_weight:g}")
        if value > t.max_value + tol:
            errors.append(f"container {k}: value {value:g} > {t.max_value:g}")
        # containment
        for it, x0, y0, z0, x1, y1, z1 in content:
            if (min(x0, y0, z0) < -tol or x1 > t.depth + tol
                    or y1 > t.width + tol or z1 > t.height + tol):
                errors.append(f"container {k}: {it.id} is not inside the container")
        # overlaps (sweep along x)
        order = sorted(content, key=lambda b: b[1])
        for a, A in enumerate(order):
            for B in order[a + 1:]:
                if B[1] >= A[4] - tol:
                    break
                if (min(A[4], B[4]) - max(A[1], B[1]) > tol
                        and min(A[5], B[5]) - max(A[2], B[2]) > tol
                        and min(A[6], B[6]) - max(A[3], B[3]) > tol):
                    errors.append(f"container {k}: {A[0].id} overlaps {B[0].id}")
        # stability
        if t.gravity > 0:
            for it, x0, y0, z0, x1, y1, z1 in content:
                if z0 <= tol:
                    continue
                base = (x1 - x0) * (y1 - y0)
                supported = sum(
                    max(0.0, min(x1, b[4]) - max(x0, b[1])) * max(0.0, min(y1, b[5]) - max(y0, b[2]))
                    for b in content if abs(b[6] - z0) <= tol
                )
                if supported + tol < t.gravity / 100.0 * base:
                    errors.append(f"container {k}: {it.id} is supported on "
                                  f"{supported / base:.0%} of its base (< {t.gravity:g}%)")
    return errors
