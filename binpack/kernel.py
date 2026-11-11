"""
Placement kernel: evaluate every (candidate point x orientation) pair for one
item in one container and return the best feasible one.

The function is written in the subset of Python that Numba can compile.
The packer and `benchmark.py` run *the same source* either interpreted by
CPython or compiled by Numba (`@njit`: single-threaded, no fastmath), so both
paths perform the same floating-point operations and take the same decisions.
"""
import numpy as np

EPS = 1e-9


def best_position(boxes, n, points, m, dims, length, width, height, support_frac):
    """
    boxes   (cap, 6) float64  placed boxes [x, y, z, dx, dy, dz]; rows 0..n-1 are used
    points  (cap, 3) float64  candidate anchor points; rows 0..m-1 are used
    dims    (k, 3)   float64  item extents (dx, dy, dz), one row per allowed orientation
    length, width, height     container extents along x, y, z
    support_frac              fraction of the item's base that must rest on the
                              top faces of other boxes (0 = no stability rule)

    Returns (point_index, orientation_index, z, contact_area);
    point_index is -1 when the item fits nowhere.

    Ranking (lexicographic): lowest z first (fill bottom-up), then the largest
    contact area with floor, walls and other boxes, then lowest x, then lowest y.
    """
    best_p = -1
    best_r = -1
    best_z = 0.0
    best_c = 0.0
    best_x = 0.0
    best_y = 0.0
    for p in range(m):
        px = points[p, 0]
        py = points[p, 1]
        pz = points[p, 2]
        for r in range(dims.shape[0]):
            dx = dims[r, 0]
            dy = dims[r, 1]
            dz = dims[r, 2]
            # 1. inside the container
            if px + dx > length + EPS or py + dy > width + EPS or pz + dz > height + EPS:
                continue
            # 2. one sweep over the placed boxes: overlap, support and contact
            feasible = True
            support = 0.0
            contact = 0.0
            for j in range(n):
                bx = boxes[j, 0]
                by = boxes[j, 1]
                bz = boxes[j, 2]
                ex = bx + boxes[j, 3]
                ey = by + boxes[j, 4]
                ez = bz + boxes[j, 5]
                ox = min(px + dx, ex) - max(px, bx)
                oy = min(py + dy, ey) - max(py, by)
                oz = min(pz + dz, ez) - max(pz, bz)
                if ox > EPS and oy > EPS and oz > EPS:
                    feasible = False
                    break
                if ox > EPS and oy > EPS:
                    if abs(ez - pz) < EPS:            # box j is right below
                        support += ox * oy
                        contact += ox * oy
                    elif abs(pz + dz - bz) < EPS:     # box j is right above
                        contact += ox * oy
                elif ox > EPS and oz > EPS:
                    if abs(ey - py) < EPS or abs(py + dy - by) < EPS:
                        contact += ox * oz
                elif oy > EPS and oz > EPS:
                    if abs(ex - px) < EPS or abs(px + dx - bx) < EPS:
                        contact += oy * oz
            if not feasible:
                continue
            # 3. stability
            if pz > EPS and support < support_frac * dx * dy - EPS:
                continue
            # floor and walls also count as contact
            if pz < EPS:
                contact += dx * dy
            if px < EPS:
                contact += dy * dz
            if py < EPS:
                contact += dx * dz
            if px + dx > length - EPS:
                contact += dy * dz
            if py + dy > width - EPS:
                contact += dx * dz
            # 4. lexicographic comparison with the incumbent
            if best_p < 0:
                better = True
            elif pz < best_z - EPS:
                better = True
            elif pz > best_z + EPS:
                better = False
            elif contact > best_c + EPS:
                better = True
            elif contact < best_c - EPS:
                better = False
            elif px < best_x - EPS:
                better = True
            elif px > best_x + EPS:
                better = False
            else:
                better = py < best_y - EPS
            if better:
                best_p = p
                best_r = r
                best_z = pz
                best_c = contact
                best_x = px
                best_y = py
    return best_p, best_r, best_z, best_c


try:
    import numba

    best_position_compiled = numba.njit(cache=True)(best_position)
    NUMBA_AVAILABLE = True
except ImportError:  # everything still works, only slower
    best_position_compiled = None
    NUMBA_AVAILABLE = False


def get_kernel(use_numba: bool = True):
    """The compiled kernel when Numba is installed (and requested), else the Python one."""
    if use_numba and NUMBA_AVAILABLE:
        return best_position_compiled
    return best_position


def warmup(kernel) -> None:
    """Trigger JIT compilation (or loading from the on-disk cache) outside any timing."""
    boxes = np.zeros((1, 6))
    points = np.zeros((1, 3))
    dims = np.ones((1, 3))
    kernel(boxes, 0, points, 1, dims, 1.0, 1.0, 1.0, 0.0)
