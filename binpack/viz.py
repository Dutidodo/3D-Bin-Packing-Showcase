"""
Drawings of a packing: static PNGs and a build-up GIF (matplotlib), and an
interactive 3D page (plotly, optional).
"""
from __future__ import annotations

import colorsys
import math
from collections import defaultdict
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from .model import oriented_extents  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
           "#DA8BC3", "#CCB974", "#64B5CD", "#2A9D8F", "#E9C46A", "#8C8C8C"]

# brightness of each face: top, bottom, y=y0, y=y1, x=x0, x=x1
_SHADE = (1.00, 0.55, 0.82, 0.70, 0.66, 0.90)


@dataclass
class Box:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    weight: float
    family: str
    label: str


@dataclass
class LoadedContainer:
    index: int
    ctype: object
    boxes: list

    @property
    def volume_fill(self) -> float:
        return sum(b.dx * b.dy * b.dz for b in self.boxes) / self.ctype.volume

    @property
    def weight_fill(self) -> float:
        return sum(b.weight for b in self.boxes) / self.ctype.max_weight

    @property
    def fill(self) -> float:
        """Binding utilisation: whichever of volume and weight is closer to its limit."""
        return max(self.volume_fill, self.weight_fill)


def _fmt(v: float) -> str:
    return f"{v:g}"


def family_of(item) -> str:
    """Synthetic items carry a family name; real ones are grouped by their sorted dimensions."""
    if item.family:
        return item.family
    return "x".join(_fmt(v) for v in sorted((item.width, item.depth, item.height)))


def containers_from_solution(instance, solution) -> list:
    """Group the placements by container, keeping the order in which they appear."""
    items = {it.id: it for it in instance.items}
    types = {t.name: t for t in instance.container_types}
    content = defaultdict(list)
    for p in solution.placements:
        it = items[p.item_id]
        dx, dy, dz = oriented_extents(it.width, it.depth, it.height, p.orient)
        label = (f"{it.id}  {_fmt(dx)}x{_fmt(dy)}x{_fmt(dz)}  "
                 f"@({_fmt(p.x)}, {_fmt(p.y)}, {_fmt(p.z)})  {_fmt(it.weight)} kg")
        content[p.container].append(Box(p.x, p.y, p.z, dx, dy, dz, it.weight, family_of(it), label))
    return [LoadedContainer(k, types[solution.containers[k]], content[k])
            for k in sorted(solution.containers)]


def color_map(containers) -> dict:
    """Stable family -> colour mapping.

    Up to 12 families use a fixed palette. With more families, hues are spread
    with the golden-ratio step, so that families next to each other in the
    ordering still get clearly different colours.
    """
    volume = {}
    for c in containers:
        for b in c.boxes:
            volume[b.family] = b.dx * b.dy * b.dz
    families = sorted(volume, key=lambda f: (-volume[f], f))
    if len(families) <= len(PALETTE):
        return {f: to_rgb(PALETTE[i]) for i, f in enumerate(families)}
    colors = {}
    for i, f in enumerate(families):
        hue = (0.58 + i * 0.618033988749895) % 1.0
        lightness = (0.50, 0.62, 0.42)[i % 3]
        colors[f] = colorsys.hls_to_rgb(hue, lightness, 0.55)
    return colors


# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------

def _faces(b: Box):
    x0, y0, z0 = b.x, b.y, b.z
    x1, y1, z1 = x0 + b.dx, y0 + b.dy, z0 + b.dz
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return [
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[0], v[1], v[5], v[4]],  # y = y0
        [v[3], v[2], v[6], v[7]],  # y = y1
        [v[0], v[3], v[7], v[4]],  # x = x0
        [v[1], v[2], v[6], v[5]],  # x = x1
    ]


def _frame(length, width, height):
    c = [(0, 0, 0), (length, 0, 0), (length, width, 0), (0, width, 0),
         (0, 0, height), (length, 0, height), (length, width, height), (0, width, height)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return [[c[a], c[b]] for a, b in edges]


def draw_container(ax, cont: LoadedContainer, colors: dict, upto=None, title=None):
    t = cont.ctype
    length, width, height = t.depth, t.width, t.height
    polys, facecolors = [], []
    for b in cont.boxes[:upto]:
        base = colors[b.family]
        for face, shade in zip(_faces(b), _SHADE):
            polys.append(face)
            facecolors.append(tuple(min(1.0, ch * shade) for ch in base))
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolors=facecolors,
                                             edgecolors=(0.12, 0.12, 0.12, 0.55),
                                             linewidths=0.35))
    ax.add_collection3d(Line3DCollection(_frame(length, width, height),
                                         colors=(0.25, 0.25, 0.25, 0.9), linewidths=0.9))
    ax.set_xlim(0, length)
    ax.set_ylim(0, width)
    ax.set_zlim(0, height)
    ax.set_box_aspect((length, width, height))
    ax.view_init(elev=24, azim=-57)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, pad=0)


def _title(cont: LoadedContainer, sep: str = "\n") -> str:
    return (f"#{cont.index} · {cont.ctype.name} · {len(cont.boxes)} items{sep}"
            f"volume {cont.volume_fill:.0%} · weight {cont.weight_fill:.0%}")


def render_container(cont: LoadedContainer, path: str, colors: dict, title=None) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=160)
    ax = fig.add_subplot(projection="3d")
    draw_container(ax, cont, colors, title=title or _title(cont))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_fleet(containers, path: str, colors: dict, cols: int = 4, suptitle=None) -> None:
    n = len(containers)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(4.4 * cols, 3.6 * rows), dpi=150)
    for i, cont in enumerate(containers):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        draw_container(ax, cont, colors, title=_title(cont))
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_build_gif(cont: LoadedContainer, path: str, colors: dict,
                     fps: int = 8, hold_seconds: float = 2.0, max_frames: int = 120) -> None:
    """Animate the container being filled, box by box, in insertion order."""
    n = len(cont.boxes)
    step = max(1, math.ceil(n / max_frames))
    frames = list(range(1, n + 1, step))
    if frames[-1] != n:
        frames.append(n)
    frames += [n] * int(hold_seconds * fps)
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(projection="3d")

    def update(k):
        ax.cla()
        draw_container(ax, cont, colors, upto=k,
                       title=f"{cont.ctype.name} #{cont.index} · {k}/{n} items")

    anim = animation.FuncAnimation(fig, update, frames=frames)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------------
# plotly (optional)
# ---------------------------------------------------------------------------

_TRIANGLES = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
              (3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 2, 6), (1, 6, 5)]
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def render_html(containers, path: str, colors: dict, title: str = "") -> bool:
    """Interactive 3D view with a menu to switch container. Returns False if plotly is missing."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False

    def rgb(c):
        return "rgb({},{},{})".format(*(int(round(255 * ch)) for ch in c))

    fig = go.Figure()
    for k, cont in enumerate(containers):
        xs, ys, zs, ii, jj, kk, fcol, text = [], [], [], [], [], [], [], []
        ex, ey, ez = [], [], []
        for b in cont.boxes:
            o = len(xs)
            x0, y0, z0 = b.x, b.y, b.z
            x1, y1, z1 = x0 + b.dx, y0 + b.dy, z0 + b.dz
            verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                     (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
            for vx, vy, vz in verts:
                xs.append(vx)
                ys.append(vy)
                zs.append(vz)
                text.append(b.label)
            for a, bb, c in _TRIANGLES:
                ii.append(o + a)
                jj.append(o + bb)
                kk.append(o + c)
                fcol.append(rgb(colors[b.family]))
            for a, bb in _EDGES:
                for v in (verts[a], verts[bb]):
                    ex.append(v[0])
                    ey.append(v[1])
                    ez.append(v[2])
                ex.append(None)
                ey.append(None)
                ez.append(None)
        for seg in _frame(cont.ctype.depth, cont.ctype.width, cont.ctype.height):
            for v in seg:
                ex.append(v[0])
                ey.append(v[1])
                ez.append(v[2])
            ex.append(None)
            ey.append(None)
            ez.append(None)
        visible = k == 0
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, facecolor=fcol,
                                flatshading=True, text=text, hoverinfo="text",
                                lighting=dict(ambient=0.65, diffuse=0.6, specular=0.05),
                                visible=visible, name=_title(cont, " · ")))
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", hoverinfo="skip",
                                   line=dict(color="rgba(30,30,30,0.6)", width=2),
                                   visible=visible, showlegend=False))
    buttons = []
    for k, cont in enumerate(containers):
        vis = [False] * (2 * len(containers))
        vis[2 * k] = vis[2 * k + 1] = True
        buttons.append(dict(label=_title(cont, " · "), method="update",
                            args=[{"visible": vis}]))
    fig.update_layout(
        title=title,
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, xanchor="left",
                          y=1.0, yanchor="top")],
        scene=dict(aspectmode="data", xaxis_title="x (depth)", yaxis_title="y (width)",
                   zaxis_title="z (height)"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.write_html(path, include_plotlyjs="cdn")
    return True
