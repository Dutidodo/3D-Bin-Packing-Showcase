"""
Problem data: container types, items, orientations and solutions, plus a
synthetic instance generator and CSV input/output in the course format.

Axes convention (same as the course problem statement):
    x = depth (length of the container), y = width, z = height.
"""
from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field

# The six orientation codes of the problem statement. Each code maps the item's
# (width, depth, height) onto the container axes. Every entry gives the indices
# into (width, depth, height) of the extent along x, along y and along z.
_ORIENTATION_AXES = {
    0: (1, 0, 2),  # no rotation
    1: (0, 1, 2),  # 90 degrees around z
    2: (1, 2, 0),  # 90 degrees around x
    3: (2, 1, 0),  # 90 degrees around x, then around z
    4: (2, 0, 1),  # 90 degrees around y
    5: (0, 2, 1),  # 90 degrees around z, then around x
}


def oriented_extents(width: float, depth: float, height: float, orient: int):
    """Extents (dx, dy, dz) along x, y, z of an item placed with `orient`."""
    dims = (width, depth, height)
    ix, iy, iz = _ORIENTATION_AXES[orient]
    return dims[ix], dims[iy], dims[iz]


@dataclass(frozen=True)
class ContainerType:
    name: str
    width: float          # extent along y
    depth: float          # extent along x
    height: float         # extent along z
    max_weight: float
    cost: float
    max_value: float = math.inf
    gravity: float = 0.0  # % of an item's base that must rest on other items

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height


@dataclass(frozen=True)
class Item:
    id: str
    width: float
    depth: float
    height: float
    weight: float
    value: float
    rotations: tuple   # allowed orientation codes
    family: str = ""   # only used to colour the drawings

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height


@dataclass(frozen=True)
class Placement:
    item_id: str
    container: int     # index of the container in the solution
    x: float
    y: float
    z: float
    orient: int


@dataclass
class Instance:
    name: str
    container_types: list
    items: list


@dataclass
class Solution:
    containers: dict                              # container index -> type name
    placements: list = field(default_factory=list)

    def cost(self, instance: Instance) -> float:
        cost_of = {t.name: t.cost for t in instance.container_types}
        return sum(cost_of[name] for name in self.containers.values())


# ---------------------------------------------------------------------------
# Synthetic instance
# ---------------------------------------------------------------------------

_DEMO_CONTAINERS = (
    ContainerType("van", width=8, depth=12, height=8, max_weight=500, cost=10, gravity=50),
    ContainerType("truck", width=10, depth=16, height=10, max_weight=1000, cost=17, gravity=60),
    ContainerType("trailer", width=12, depth=24, height=10, max_weight=1800, cost=30, gravity=75),
)

# family, (width, depth, height), allowed orientations, weight per unit volume, frequency
_DEMO_FAMILIES = (
    ("crate", (4, 6, 4), "012345", 0.20, 3),
    ("carton", (4, 4, 3), "012345", 0.12, 4),
    ("flat", (6, 6, 2), "01", 0.15, 2),
    ("tall", (3, 3, 6), "0", 0.18, 2),       # must stay upright
    ("long", (2, 8, 2), "012345", 0.10, 2),
    ("cube", (5, 5, 5), "012345", 0.16, 1),
    ("small", (2, 3, 2), "012345", 0.25, 4),
    ("heavy", (4, 4, 2), "01", 0.90, 1),
)


def make_synthetic_instance(seed: int = 7, n_items: int = 120) -> Instance:
    """A reproducible heterogeneous instance: 3 container types, 8 box families."""
    rng = random.Random(seed)
    weights = [f[4] for f in _DEMO_FAMILIES]
    items = []
    for i in range(n_items):
        name, (w, d, h), rots, density, _ = rng.choices(_DEMO_FAMILIES, weights=weights)[0]
        weight = round(w * d * h * density * rng.uniform(0.8, 1.2), 1)
        value = round(weight * rng.uniform(0.5, 2.0), 1)
        items.append(Item(f"I{i:03d}", float(w), float(d), float(h), weight, value,
                          tuple(int(c) for c in rots), family=name))
    return Instance(f"synthetic-seed{seed}", list(_DEMO_CONTAINERS), items)


# ---------------------------------------------------------------------------
# CSV input/output (course format)
# ---------------------------------------------------------------------------

def _float_or(text, default: float) -> float:
    if text is None or str(text).strip() == "":
        return default
    value = float(text)
    return default if math.isnan(value) else value


def load_course_instance(dataset_dir: str) -> Instance:
    """Read `vehicles.csv` and `items.csv` from a dataset folder."""
    with open(os.path.join(dataset_dir, "vehicles.csv"), newline="", encoding="utf-8") as f:
        types = [
            ContainerType(
                name=r["type"],
                width=float(r["width"]),
                depth=float(r["depth"]),
                height=float(r["height"]),
                max_weight=float(r["maxWeight"]),
                cost=float(r["cost"]),
                max_value=_float_or(r.get("maxValue"), math.inf),
                gravity=_float_or(r.get("gravityStrength"), 0.0),
            )
            for r in csv.DictReader(f)
        ]
    with open(os.path.join(dataset_dir, "items.csv"), newline="", encoding="utf-8") as f:
        items = [
            Item(
                id=r["id"],
                width=float(r["width"]),
                depth=float(r["depth"]),
                height=float(r["height"]),
                weight=float(r["weight"]),
                value=float(r["value"]),
                rotations=tuple(int(c) for c in r["allowedRotations"].strip()),
            )
            for r in csv.DictReader(f)
        ]
    return Instance(os.path.basename(os.path.normpath(dataset_dir)), types, items)


def load_course_solution(path: str) -> Solution:
    """Read a solution CSV (type_vehicle, idx_vehicle, id_item, x/y/z_origin, orient)."""
    containers, placements = {}, []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = int(r["idx_vehicle"])
            if containers.setdefault(k, r["type_vehicle"]) != r["type_vehicle"]:
                raise ValueError(f"container {k} appears with two different types")
            placements.append(Placement(r["id_item"], k, float(r["x_origin"]),
                                        float(r["y_origin"]), float(r["z_origin"]),
                                        int(r["orient"])))
    return Solution(containers, placements)


def save_solution_csv(solution: Solution, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["type_vehicle", "idx_vehicle", "id_item",
                      "x_origin", "y_origin", "z_origin", "orient"])
        for p in solution.placements:
            out.writerow([solution.containers[p.container], p.container, p.item_id,
                          p.x, p.y, p.z, p.orient])
