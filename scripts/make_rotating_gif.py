"""
Render a rotating GIF from a colored .obj mesh using a pure NumPy software
rasterizer (no OpenGL / no display required).

Usage:
    python3 scripts/make_rotating_gif.py \
        --obj archicmeshvis/000020_combined_pose.obj \
        --out assets/rotating_mesh.gif \
        --size 480 --frames 36 --fps 18
"""
import argparse
import os
import numpy as np
from PIL import Image
import trimesh


def load_mesh(path):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    if m.visual.kind == "vertex" and m.visual.vertex_colors is not None:
        C = np.asarray(m.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
    else:
        C = np.full_like(V, fill_value=0.4)
        C[:, 2] = 0.9  # default blue
    return V, F, C


def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rasterize(V, F, C, size, bg=(35, 35, 38)):
    """Software z-buffer rasterizer with per-vertex color + Lambert shading."""
    H = W = size
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = bg[0] / 255.0
    img[..., 1] = bg[1] / 255.0
    img[..., 2] = bg[2] / 255.0
    zbuf = np.full((H, W), -np.inf, dtype=np.float32)

    # Fit mesh into NDC roughly
    center = (V.max(0) + V.min(0)) / 2
    scale = 0.9 / np.max(V.max(0) - V.min(0))
    P = (V - center) * scale  # in [-0.45, 0.45]
    sx = (P[:, 0] * 0.5 + 0.5) * (W - 1)
    sy = (1.0 - (P[:, 1] * 0.5 + 0.5)) * (H - 1)
    sz = P[:, 2]

    # Compute per-face normals for shading, and backface culling
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n /= nlen
    light = np.array([0.3, 0.4, 1.0])
    light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.0, 1.0) * 0.7 + 0.3  # ambient + diffuse

    # Per-face vertex screen coords
    x0, x1, x2 = sx[F[:, 0]], sx[F[:, 1]], sx[F[:, 2]]
    y0, y1, y2 = sy[F[:, 0]], sy[F[:, 1]], sy[F[:, 2]]
    z0, z1, z2 = sz[F[:, 0]], sz[F[:, 1]], sz[F[:, 2]]
    c0, c1, c2 = C[F[:, 0]], C[F[:, 1]], C[F[:, 2]]

    # Sort faces back-to-front for nicer overlap (we still z-buffer)
    zmean = (z0 + z1 + z2) / 3.0
    order = np.argsort(zmean)

    for i in order:
        ax, ay, az = x0[i], y0[i], z0[i]
        bx, by, bz = x1[i], y1[i], z1[i]
        cx, cy, cz = x2[i], y2[i], z2[i]
        ca, cb, cc = c0[i], c1[i], c2[i]
        s = shade[i]

        xmin = max(int(np.floor(min(ax, bx, cx))), 0)
        xmax = min(int(np.ceil(max(ax, bx, cx))), W - 1)
        ymin = max(int(np.floor(min(ay, by, cy))), 0)
        ymax = min(int(np.ceil(max(ay, by, cy))), H - 1)
        if xmax < xmin or ymax < ymin:
            continue

        ys, xs = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)

        denom = ((by - cy) * (ax - cx) + (cx - bx) * (ay - cy))
        if abs(denom) < 1e-8:
            continue
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / denom
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            continue

        zs = w0 * az + w1 * bz + w2 * cz
        sub_z = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        write = mask & (zs > sub_z)
        if not write.any():
            continue

        col = (w0[..., None] * ca + w1[..., None] * cb + w2[..., None] * cc) * s
        sub_img = img[ymin:ymax + 1, xmin:xmax + 1]
        sub_img[write] = col[write]
        sub_z[write] = zs[write]

    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--fps", type=int, default=18)
    args = ap.parse_args()

    V, F, C = load_mesh(args.obj)
    print(f"Loaded: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Pre-center and put on Y-up. Many MANO/Arctic exports are Y-down.
    V = V - V.mean(0)
    V[:, 1] *= -1  # flip Y for upright view
    V[:, 2] *= -1

    images = []
    for k in range(args.frames):
        theta = 2 * np.pi * k / args.frames
        Vr = V @ rotation_y(theta).T
        frame = rasterize(Vr, F, C, args.size)
        images.append(Image.fromarray(frame))
        print(f"  frame {k + 1}/{args.frames}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    duration_ms = int(1000 / args.fps)
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
