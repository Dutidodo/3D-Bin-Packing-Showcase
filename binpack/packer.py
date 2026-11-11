"""
Constructive heuristic on extreme points plus a small local search.

Construction: items are taken in a given order; each one goes into the first
open container that can host it, at the best (point, orientation) found by the
kernel; if none can, a new container is opened, choosing the type with the
lowest cost per unit of volume that can host the item.

Local search (strict improvement only, until no move applies):
  * eliminate  - empty the least-filled container into the others;
  * downgrade  - repack one container into a cheaper container type;
  * repack     - pool the items of two sparse containers and rebuild them from
                 scratch into a strictly cheaper set of one or two containers.
"""
from __future__ import annotations

import itertools
import math
import random
import time

import numpy as np

from .kernel import EPS, get_kernel, warmup
from .model import Instance, Placement, Solution, oriented_extents


def _key(x: float, y: float, z: float):
    return (round(x, 6), round(y, 6), round(z, 6))


class Bin:
    """One open container: placed boxes and candidate (extreme) points as NumPy arrays."""

    def __init__(self, ctype):
        self.ctype = ctype
        self.boxes = np.zeros((32, 6))   # [x, y, z, dx, dy, dz]
        self.n = 0
        self.points = np.zeros((64, 3))  # candidate anchor points; the origin first
        self.m = 1
        self._point_keys = {_key(0.0, 0.0, 0.0)}
        self.weight = 0.0
        self.value = 0.0
        self.volume = 0.0
        self.placements = []             # (item, orient, x, y, z) in insertion order

    def copy(self) -> "Bin":
        b = Bin.__new__(Bin)
        b.ctype = self.ctype
        b.boxes = self.boxes.copy()
        b.n = self.n
        b.points = self.points.copy()
        b.m = self.m
        b._point_keys = set(self._point_keys)
        b.weight, b.value, b.volume = self.weight, self.value, self.volume
        b.placements = list(self.placements)
        return b

    @property
    def cost(self) -> float:
        return self.ctype.cost

    def items(self):
        return [p[0] for p in self.placements]

    def fill(self) -> float:
        """Binding utilisation: the largest of volume, weight and value fill ratios."""
        t = self.ctype
        f = self.volume / t.volume
        if t.max_weight > 0:
            f = max(f, self.weight / t.max_weight)
        if math.isfinite(t.max_value) and t.max_value > 0:
            f = max(f, self.value / t.max_value)
        return f

    def has_capacity_for(self, item) -> bool:
        return (self.weight + item.weight <= self.ctype.max_weight + EPS
                and self.value + item.value <= self.ctype.max_value + EPS)

    def add(self, item, orient, x, y, z, dx, dy, dz) -> None:
        if self.n == self.boxes.shape[0]:
            self.boxes = np.vstack([self.boxes, np.zeros_like(self.boxes)])
        self.boxes[self.n] = (x, y, z, dx, dy, dz)
        self.n += 1
        self.weight += item.weight
        self.value += item.value
        self.volume += dx * dy * dz
        self.placements.append((item, orient, x, y, z))
        self._update_points(x, y, z, dx, dy, dz)

    # -- extreme points ------------------------------------------------------

    def _update_points(self, x, y, z, dx, dy, dz) -> None:
        # 1. drop the candidate points now covered by the new box
        pts = self.points[:self.m]
        inside = ((pts[:, 0] >= x - EPS) & (pts[:, 0] < x + dx - EPS)
                  & (pts[:, 1] >= y - EPS) & (pts[:, 1] < y + dy - EPS)
                  & (pts[:, 2] >= z - EPS) & (pts[:, 2] < z + dz - EPS))
        if inside.any():
            for px, py, pz in pts[inside]:
                self._point_keys.discard(_key(px, py, pz))
            kept = pts[~inside]
            self.m = len(kept)
            self.points[:self.m] = kept
        # 2. the three corners of the new box, each also projected back along
        #    the two other axes until it meets a wall or another box
        for corner, axes in (((x + dx, y, z), (1, 2)),
                             ((x, y + dy, z), (0, 2)),
                             ((x, y, z + dz), (0, 1))):
            self._add_point(*corner)
            for axis in axes:
                self._add_point(*self._project(corner, axis))

    def _project(self, point, axis):
        p = list(point)
        a1, a2 = [a for a in (0, 1, 2) if a != axis]
        b = self.boxes[:self.n]
        end = b[:, axis] + b[:, axis + 3]
        hit = ((end <= p[axis] + EPS)
               & (b[:, a1] - EPS <= p[a1]) & (p[a1] < b[:, a1] + b[:, a1 + 3] - EPS)
               & (b[:, a2] - EPS <= p[a2]) & (p[a2] < b[:, a2] + b[:, a2 + 3] - EPS))
        p[axis] = float(end[hit].max()) if hit.any() else 0.0
        return p

    def _add_point(self, px, py, pz) -> None:
        t = self.ctype
        if px >= t.depth - EPS or py >= t.width - EPS or pz >= t.height - EPS:
            return
        k = _key(px, py, pz)
        if k in self._point_keys:
            return
        if self.m == self.points.shape[0]:
            self.points = np.vstack([self.points, np.zeros_like(self.points)])
        self.points[self.m] = (px, py, pz)
        self.m += 1
        self._point_keys.add(k)


def total_cost(bins) -> float:
    return sum(b.cost for b in bins)


class Packer:
    def __init__(self, instance: Instance, use_numba: bool = True):
        self.instance = instance
        self.kernel = get_kernel(use_numba)
        # allowed orientation codes and the matching extents, once per item
        self._orients = {}
        for it in instance.items:
            codes = list(it.rotations)
            dims = np.array([oriented_extents(it.width, it.depth, it.height, o) for o in codes],
                            dtype=np.float64).reshape(-1, 3)
            self._orients[it.id] = (codes, dims)
        types = instance.container_types
        self.types_by_efficiency = sorted(types, key=lambda t: (t.cost / t.volume, t.cost))
        self.types_by_cost = sorted(types, key=lambda t: (t.cost, -t.volume))

    def warmup(self) -> float:
        """Compile (or load) the kernel; returns the seconds it took."""
        t0 = time.perf_counter()
        warmup(self.kernel)
        return time.perf_counter() - t0

    # -- single moves ----------------------------------------------------------

    def try_insert(self, b: Bin, item) -> bool:
        if not b.has_capacity_for(item):
            return False
        codes, dims = self._orients[item.id]
        t = b.ctype
        p, r, _, _ = self.kernel(b.boxes, b.n, b.points, b.m, dims,
                                 float(t.depth), float(t.width), float(t.height),
                                 t.gravity / 100.0)
        if p < 0:
            return False
        x, y, z = (float(v) for v in b.points[p])
        dx, dy, dz = (float(v) for v in dims[r])
        b.add(item, codes[r], x, y, z, dx, dy, dz)
        return True

    def insert_first_fit(self, bins, item) -> bool:
        return any(self.try_insert(b, item) for b in bins)

    def open_bin(self, item) -> Bin:
        for t in self.types_by_efficiency:
            b = Bin(t)
            if self.try_insert(b, item):
                return b
        raise ValueError(f"item {item.id} does not fit in any container type")

    def construct(self, order) -> list:
        bins = []
        for item in order:
            if not self.insert_first_fit(bins, item):
                bins.append(self.open_bin(item))
        return bins

    def pack_fresh(self, items, types):
        """Pack `items` into fresh containers of the given types; None if they do not all fit."""
        bins = [Bin(t) for t in types]
        for item in items:
            if not self.insert_first_fit(bins, item):
                return None
        return [b for b in bins if b.n > 0]

    # -- local search moves (each returns a strictly cheaper list, or None) ---

    def eliminate(self, bins):
        for target in sorted(bins, key=lambda b: (b.fill(), -b.cost)):
            others = [b.copy() for b in bins if b is not target]
            items = sorted(target.items(), key=lambda it: -it.volume)
            if all(self.insert_first_fit(others, it) for it in items):
                return others
        return None

    def downgrade(self, bins):
        for i, b in sorted(enumerate(bins), key=lambda e: -e[1].cost):
            items = sorted(b.items(), key=lambda it: -it.volume)
            for t in self.types_by_cost:
                if t.cost >= b.cost - EPS:
                    break
                if (b.volume > t.volume + EPS or b.weight > t.max_weight + EPS
                        or b.value > t.max_value + EPS):
                    continue
                fresh = self.pack_fresh(items, [t])
                if fresh is not None and len(fresh) == 1:
                    return bins[:i] + fresh + bins[i + 1:]
        return None

    def repack(self, bins):
        sparse = sorted(bins, key=lambda b: b.fill())[:4]
        for b1, b2 in itertools.combinations(sparse, 2):
            old = b1.cost + b2.cost
            items = sorted(b1.items() + b2.items(), key=lambda it: -it.volume)
            vol = sum(it.volume for it in items)
            wt = sum(it.weight for it in items)
            val = sum(it.value for it in items)
            options = []
            for k in (1, 2):
                for combo in itertools.combinations_with_replacement(self.types_by_cost, k):
                    cost = sum(t.cost for t in combo)
                    if (cost < old - EPS and sum(t.volume for t in combo) >= vol - EPS
                            and sum(t.max_weight for t in combo) >= wt - EPS
                            and sum(t.max_value for t in combo) >= val - EPS):
                        options.append((cost, sorted(combo, key=lambda t: -t.volume)))
            for _, combo in sorted(options, key=lambda o: o[0]):
                fresh = self.pack_fresh(items, combo)
                if fresh is not None:
                    return [b for b in bins if b is not b1 and b is not b2] + fresh
        return None

    def improve(self, bins, deadline=None):
        while deadline is None or time.perf_counter() < deadline:
            for move in (self.eliminate, self.downgrade, self.repack):
                new = move(bins)
                if new is not None and total_cost(new) < total_cost(bins) - EPS:
                    bins = new
                    break
            else:
                break  # no move improves: local optimum
        return bins

    # -- full run ----------------------------------------------------------------

    def orderings(self, restarts: int, seed: int):
        items = list(self.instance.items)
        yield "volume", sorted(items, key=lambda it: (-it.volume, it.id))
        yield "base area", sorted(items, key=lambda it: (-it.width * it.depth, -it.height, it.id))
        yield "height", sorted(items, key=lambda it: (-it.height, -it.volume, it.id))
        rng = random.Random(seed)
        for k in range(restarts):
            noise = {it.id: rng.uniform(0.6, 1.4) for it in items}
            yield f"noisy volume #{k + 1}", sorted(items, key=lambda it: (-it.volume * noise[it.id], it.id))

    def solve(self, restarts: int = 6, seed: int = 0, time_limit=None):
        """Multi-start: construct with several orderings, improve each, keep the best.

        Deterministic for a given seed unless a time limit is given.
        Returns (best bins, history rows (ordering, constructive cost, improved cost, #containers)).
        """
        deadline = None if time_limit is None else time.perf_counter() + time_limit
        best, history = None, []
        for name, order in self.orderings(restarts, seed):
            bins = self.construct(order)
            c0 = total_cost(bins)
            bins = self.improve(bins, deadline)
            history.append((name, c0, total_cost(bins), len(bins)))
            if best is None or total_cost(bins) < total_cost(best) - EPS:
                best = bins
            if deadline is not None and time.perf_counter() >= deadline:
                break
        return best, history

    def to_solution(self, bins) -> Solution:
        containers, placements = {}, []
        for k, b in enumerate(bins):
            containers[k] = b.ctype.name
            for item, orient, x, y, z in b.placements:
                placements.append(Placement(item.id, k, x, y, z, orient))
        return Solution(containers, placements)
