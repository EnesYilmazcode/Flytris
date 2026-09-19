"""Pose the NeuroMechFly body (micro-CT Drosophila, flygym 2.1.0, Apache-2.0) for the arcade.

The fly sits on a bar stool with its body pitched up, mid and hind feet on the seat,
left front foot on the joystick and right front foot on the first button. Leg angles
come from least-squares IK starting at flygym's neutral pose.

Writes fly_model.json: hero parts at full mesh detail, a decimated crowd mesh, wing
rest and flight poses, and the pivot/axis the death animation needs. Units are fly
units (body length about 1), frame: origin on the seat top, -z forward, +y up.
"""
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
NMF = HERE / "nmf"
NMF_FULL = HERE / "nmf_full"  # full-resolution meshes from flygym's public asset bucket, hero only
MM = 1 / 2.7            # fly units per mm
PITCH = np.radians(16)  # nose up
THORAX_AT = np.array([0.0, 0.40, -0.26])  # thorax origin in the fly frame

# Cabinet geometry from cabinet.js, converted to the fly frame (world / FLY_SCALE).
FLY_SCALE, SEAT_Y, STOOL_Z = 1.35, 0.72, 1.55
PANEL = np.arctan2(0.12, 0.36)


def world_to_fly(x, y, z):
    return np.array([x, y - SEAT_Y, z - STOOL_Z]) / FLY_SCALE


def panel_point(x, t, lift):
    p = np.array([x, 1.18 + 0.12 * t, 0.66 - 0.36 * t])
    return p + lift * np.array([0, np.cos(PANEL), np.sin(PANEL)])


JOY_LEN, JOY_TILT = 0.13, np.radians(18)
BUTTON_X = [0.06, 0.18, 0.3]


def joystick_target(tilt):
    """Ball top with the stick tilted; tilt +1 leans right (+x)."""
    base = panel_point(-0.22, 0.4, 0.0)
    n = np.array([0, np.cos(PANEL), np.sin(PANEL)])
    a = -tilt * JOY_TILT  # about the panel's forward axis; positive angles lean the ball to -x
    ball = base + JOY_LEN * (n * np.cos(a) + np.array([-1.0, 0, 0]) * np.sin(a))
    return world_to_fly(*(ball + [0, 0.04, 0]))


def button_target(b, lift):
    return world_to_fly(*panel_point(BUTTON_X[b], 0.45, lift))


HOVER, PRESS = 0.085, 0.036
FEET = {
    "lf": joystick_target(0),
    "rf": button_target(0, 0.045),
    "lm": np.array([-0.24, 0.0, -0.12]), "rm": np.array([0.24, 0.0, -0.12]),
    "lh": np.array([-0.2, 0.0, 0.16]), "rh": np.array([0.2, 0.0, 0.16]),
}

SIDES = ["l", "r"]
LEGS = [s + p for s in SIDES for p in "fmh"]
LEG_LINKS = ["coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5"]
ABDOMEN = ["c_abdomen12", "c_abdomen3", "c_abdomen4", "c_abdomen5", "c_abdomen6"]

PARENT = {"c_head": "c_thorax", "c_rostrum": "c_head", "c_haustellum": "c_rostrum"}
prev = "c_thorax"
for a in ABDOMEN:
    PARENT[a], prev = prev, a
for s in SIDES:
    PARENT.update({f"{s}_eye": "c_head", f"{s}_pedicel": "c_head", f"{s}_funiculus": f"{s}_pedicel",
                   f"{s}_arista": f"{s}_funiculus", f"{s}_wing": "c_thorax", f"{s}_haltere": "c_thorax"})
for leg in LEGS:
    prev = "c_thorax"
    for lk in LEG_LINKS:
        PARENT[f"{leg}_{lk}"], prev = prev, f"{leg}_{lk}"
ORDER = ["c_thorax"] + [k for k in PARENT]  # parents always precede children above

RIG = yaml.safe_load((NMF / "rigging.yaml").read_text())
NEUTRAL = yaml.safe_load((NMF / "yaw_pitch_roll.yaml").read_text())["joint_angles"]
AXES = {"yaw": np.array([1.0, 0, 0]), "pitch": np.array([0, 1.0, 0]), "roll": np.array([0, 0, 1.0])}
IK_DOFS = [("coxa", "yaw"), ("coxa", "pitch"), ("coxa", "roll"), ("trochanterfemur", "pitch"),
           ("trochanterfemur", "roll"), ("tibia", "pitch"), ("tarsus1", "pitch")]


def mirror_name(name):
    return name.replace("c_thorax-l", "c_thorax-r").replace("-l", "-r").replace("l_", "r_", 1) \
        if name.startswith(("c_thorax-l", "l")) else name


def full_neutral():
    angles = {}
    for name, deg in NEUTRAL.items():
        angles[name] = np.radians(deg)
        parent, child, axis = name.split("-")
        rname = f"{parent.replace('lf', 'rf').replace('lm', 'rm').replace('lh', 'rh').replace('l_', 'r_')}-" \
                f"{child.replace('lf', 'rf').replace('lm', 'rm').replace('lh', 'rh').replace('l_', 'r_')}-{axis}"
        angles[rname] = np.radians(deg)
    return angles


def joint_rot(child, angles):
    parent = PARENT[child]
    rot = np.eye(3)
    for axis in ["yaw", "pitch", "roll"]:
        a = angles.get(f"{parent}-{child}-{axis}", 0.0)
        if not a:
            continue
        vec = AXES[axis] * (-1 if child[0] == "r" and axis != "pitch" else 1)
        rot = rot @ Rotation.from_rotvec(vec * a).as_matrix()
    return rot


def local(child, angles, quat_override=None):
    m = np.eye(4)
    m[:3, 3] = RIG[child]["pos"]
    w, x, y, z = quat_override if quat_override is not None else RIG[child]["quat"]
    m[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix() @ joint_rot(child, angles)
    return m


def world_transforms(angles, overrides=None):
    overrides = overrides or {}
    T = {"c_thorax": np.eye(4)}
    for seg in ORDER[1:]:
        T[seg] = T[PARENT[seg]] @ local(seg, angles, overrides.get(seg))
    return T


# NeuroMechFly body frame (x forward, y left, z up, mm) -> fly frame.
BASIS = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float)  # fly x = -y, fly y = z, fly z = -x
PITCH_ROT = Rotation.from_rotvec([0, -PITCH, 0]).as_matrix()   # nose up about the NMF y axis


def to_fly(p_mm):
    return (BASIS @ (PITCH_ROT @ np.asarray(p_mm).T)).T * MM + THORAX_AT


def load_mesh(seg, full=False):
    src = seg if seg[0] != "r" else "l" + seg[1:]
    path = NMF_FULL / f"{src}.stl" if full and (NMF_FULL / f"{src}.stl").exists() else NMF / f"{src}.stl"
    mesh = trimesh.load(path)
    if full and len(mesh.faces) < 8000:
        try:
            mesh = mesh.subdivide_loop(iterations=1)
        except Exception as e:  # non-manifold pieces stay as they are
            print("no subdivision for", seg, e)
    v = np.asarray(mesh.vertices) * 1000.0
    f = np.asarray(mesh.faces)
    if seg[0] == "r":
        v = v * [1, -1, 1]
        f = f[:, ::-1]
    return v, f


MESHES = {seg: load_mesh(seg) for seg in ORDER}
TIP = {leg: MESHES[f"{leg}_tarsus5"][0][np.argmin(MESHES[f"{leg}_tarsus5"][0][:, 2])] for leg in LEGS}


def foot(leg, angles):
    T = world_transforms(angles)[f"{leg}_tarsus5"]
    return to_fly((T @ np.append(TIP[leg], 1))[:3])


def knee_pos(leg, angles):
    return to_fly(world_transforms(angles)[f"{leg}_tibia"][:3, 3])


def knee_goal(leg, angles, target):
    # Elbow out to the side and dropped below the reach line, so the front legs read as arms.
    attach = to_fly(world_transforms(angles)[f"{leg}_coxa"][:3, 3])
    side = -1 if leg[0] == "l" else 1
    mid = (attach + target) / 2
    return mid + np.array([0.13 * side, -0.14, 0.04])


def leg_dofs(leg):
    return [f"{PARENT[f'{leg}_{lk}']}-{leg}_{lk}-{ax}" for lk, ax in IK_DOFS]


def solve_leg(leg, angles, target, anchor=None):
    """Least-squares IK for one leg; anchor keeps nearby poses close to each other."""
    names = leg_dofs(leg)
    x0 = np.array([angles.get(n, 0.0) for n in names])
    ref = x0 if anchor is None else anchor
    goal = knee_goal(leg, angles, target) if leg[1] == "f" else None

    def resid(x):
        a = dict(angles)
        a.update(zip(names, x))
        r = [(foot(leg, a) - target) * 40, (x - ref) * 0.15]
        if goal is not None:
            r.append((knee_pos(leg, a) - goal) * 8)
        return np.concatenate(r)

    sol = least_squares(resid, x0, bounds=(x0 - 2.2, x0 + 2.2))
    out = dict(angles)
    out.update(zip(names, sol.x))
    return out, float(np.linalg.norm(foot(leg, out) - target))


def solve_legs():
    angles = full_neutral()
    for leg in LEGS:
        angles, err = solve_leg(leg, angles, FEET[leg])
        print(f"{leg}: foot error {err:.3f} fly units")
    return angles


def front_rig(angles):
    """Joint angles for the front legs at every hand position the video needs."""
    poses = {}
    rest_l = np.array([angles[n] for n in leg_dofs("lf")])
    for tilt in (-1, 0, 1):
        a, err = solve_leg("lf", angles, joystick_target(tilt), anchor=rest_l)
        poses[f"joy{tilt:+d}"] = a
        print(f"lf joystick {tilt:+d}: error {err:.3f}")
    rest_r = np.array([angles[n] for n in leg_dofs("rf")])
    for b in range(3):
        for name, lift in (("hover", HOVER), ("press", PRESS)):
            a, err = solve_leg("rf", angles, button_target(b, lift), anchor=rest_r)
            poses[f"b{b}_{name}"] = a
            print(f"rf button {b} {name}: error {err:.3f}")
    return poses


def colorize(seg, v_local):
    """Per-vertex linear RGB from the segment name and its local coordinates."""
    n = len(v_local)
    c = lambda rgb: np.tile(np.array(rgb, float) ** 2.2, (n, 1))
    if seg.endswith("_eye"):
        return c([0.62, 0.09, 0.07])
    if seg.endswith("_wing"):
        return c([0.85, 0.85, 0.88])
    if seg.endswith("_arista") or seg.endswith("tarsus5"):
        return c([0.22, 0.14, 0.07])
    if "tarsus" in seg:
        return c([0.52, 0.36, 0.17])
    if seg.startswith("c_abdomen"):
        x, z = v_local[:, 0], v_local[:, 2]
        xmin = x.min()
        dark = (x < xmin * 0.55) | (seg in ("c_abdomen5", "c_abdomen6"))
        dorsal = np.clip((z - z.min()) / (np.ptp(z) + 1e-9) * 2.2 - 0.6, 0, 1)
        base = np.array([0.8, 0.64, 0.42]) ** 2.2
        band = np.array([0.2, 0.12, 0.05]) ** 2.2
        k = (dark * dorsal * 0.95)[:, None]
        return base * (1 - k) + band * k
    if seg == "c_thorax":
        z = v_local[:, 2]
        k = np.clip((z - z.min()) / np.ptp(z), 0, 1)[:, None]
        return (np.array([0.62, 0.45, 0.24]) ** 2.2) * (1 - k) + (np.array([0.5, 0.36, 0.2]) ** 2.2) * k
    if seg == "c_head":
        return c([0.72, 0.52, 0.28])
    return c([0.64, 0.46, 0.22])


def part_category(seg):
    if seg.endswith("_eye"):
        return "eye"
    if seg.endswith("_wing"):
        return "wing"
    return "body"


def build(angles, meshes):
    T = world_transforms(angles)
    parts = []
    for seg in ORDER:
        v, f = meshes[seg]
        vw = (T[seg] @ np.c_[v, np.ones(len(v))].T).T[:, :3]
        parts.append(dict(seg=seg, cat=part_category(seg), v=to_fly(vw), f=f, col=colorize(seg, v), vloc=v))
    return parts, T


def wing_flight(angles, side):
    """Wings spread sideways and slightly raised, for take-off."""
    seg = f"{side}_wing"
    sign = 1 if side == "l" else -1
    q = Rotation.from_euler("zx", [sign * 12, -sign * 8], degrees=True).as_quat()  # x, y, z, w
    T = world_transforms(angles, {seg: [q[3], q[0], q[1], q[2]]})
    v = MESHES[seg][0]
    return to_fly((T[seg] @ np.c_[v, np.ones(len(v))].T).T[:, :3]), to_fly(T[seg][:3, 3])


def wing_uv(v_local):
    # The wing mesh runs along local +y (root to tip); x spans the chord.
    y, x = v_local[:, 1], v_local[:, 0]
    u = (y - y.min()) / np.ptp(y)
    w = (x - x.min()) / np.ptp(x)
    return np.c_[u, w]


def bristles(parts, rng):
    """Dorsal macrochaetae on thorax and head as thin cones, pointing back."""
    out_v, out_f = [], []
    for p in parts:
        if p["seg"] not in ("c_thorax", "c_head"):
            continue
        mesh = trimesh.Trimesh(p["v"], p["f"], process=False)
        normals = mesh.vertex_normals
        up = normals[:, 1] > (0.75 if p["seg"] == "c_thorax" else 0.6)
        idx = np.nonzero(up)[0]
        count = 28 if p["seg"] == "c_thorax" else 10
        for i in rng.choice(idx, size=min(count, len(idx)), replace=False):
            base = mesh.vertices[i]
            d = normals[i] * 0.35 + np.array([np.sign(base[0]) * 0.2, 0.25, 1.0])
            d /= np.linalg.norm(d)
            length = rng.uniform(0.05, 0.1)
            cone = trimesh.creation.cone(radius=0.0045, height=length, sections=5)
            align = trimesh.geometry.align_vectors([0, 0, 1], d)
            cone.apply_transform(align)
            cone.apply_translation(base - d * 0.004)
            out_f.append(cone.faces + sum(len(x) for x in out_v))
            out_v.append(cone.vertices)
    return np.vstack(out_v), np.vstack(out_f)


BLOB = bytearray()


def ref(a, dtype):
    """Append an array to the binary blob and return its descriptor."""
    a = np.ascontiguousarray(a, dtype=dtype).ravel()
    while len(BLOB) % 4:
        BLOB.append(0)
    d = dict(offset=len(BLOB), length=int(a.size), type=np.dtype(dtype).name)
    BLOB.extend(a.tobytes())
    return d


def pack(v, f, col=None, extra=None):
    d = dict(position=ref(v, np.float32), index=ref(f, np.uint32))
    if col is not None:
        d["color"] = ref(col, np.float32)
    if extra:
        d.update(extra)
    return d


def merge(items):
    vs, fs, cs, off = [], [], [], 0
    for v, f, c in items:
        vs.append(v)
        fs.append(f + off)
        cs.append(c)
        off += len(v)
    return np.vstack(vs), np.vstack(fs), np.vstack(cs)


def main():
    angles = solve_legs()
    poses = front_rig(angles)
    parts, T = build(angles, MESHES)
    full = {seg: load_mesh(seg, full=True) for seg in ORDER}
    hero_parts, _ = build(angles, full)
    rng = np.random.default_rng(5)
    hero = {}
    # Hero front legs are animated in the browser, so they leave the static body mesh.
    animated = lambda seg: seg.startswith(("lf_", "rf_"))
    for cat in ["body", "eye"]:
        v, f, c = merge([(p["v"], p["f"], p["col"]) for p in hero_parts if p["cat"] == cat and not animated(p["seg"])])
        hero[cat] = pack(v, f, c)

    # Rig: per segment, its rest transform, joint axes and full-res mesh in local mm.
    root = np.eye(4)
    root[:3, :3] = MM * BASIS @ PITCH_ROT
    root[:3, 3] = THORAX_AT
    rig = dict(root=root.T.ravel().tolist(), legs={})
    for leg in ("lf", "rf"):
        segs = []
        for lk in LEG_LINKS:
            seg = f"{leg}_{lk}"
            rest = np.eye(4)
            rest[:3, 3] = RIG[seg]["pos"]
            w, x, y, z = RIG[seg]["quat"]
            rest[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            axes = [(AXES[ax] * (-1 if seg[0] == "r" and ax != "pitch" else 1)).tolist() for ax in ("yaw", "pitch", "roll")]
            v, f = full[seg]
            segs.append(dict(name=seg, rest=rest.T.ravel().tolist(), axes=axes,
                             mesh=pack(v, f, colorize(seg, v))))
        pose_angles = {
            name: [[float(a.get(f"{PARENT[sg['name']]}-{sg['name']}-{ax}", 0.0)) for ax in ("yaw", "pitch", "roll")]
                   for sg in segs]
            for name, a in poses.items() if name.startswith("joy" if leg == "lf" else "b")
        }
        rig["legs"][leg] = dict(segments=segs, poses=pose_angles)
    bv, bf = bristles(parts, rng)
    hero["bristle"] = pack(bv, bf, np.tile(np.array([0.05, 0.035, 0.025]), (len(bv), 1)))

    wings = {}
    for side in SIDES:
        p = next(x for x in parts if x["seg"] == f"{side}_wing")
        flight, hinge = wing_flight(angles, side)
        wings[side] = pack(p["v"], p["f"], extra=dict(
            flight=ref(flight, np.float32), uv=ref(wing_uv(p["vloc"]), np.float32),
            hinge=hinge.tolist(), side=-1 if side == "l" else 1))

    # Crowd: one decimated mesh for body + eyes (wings are shared with the hero).
    v, f, c = merge([(p["v"], p["f"], p["col"]) for p in parts if p["cat"] != "wing"])
    import fast_simplification
    vs, fs = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=0.96)
    nearest = trimesh.Trimesh(v, f, process=False).kdtree.query(vs)[1]
    crowd = pack(vs, fs, c[nearest])

    thorax = to_fly([-0.45, 0, 0])  # middle of the thorax, the take-off pivot
    hv = next(p["v"] for p in parts if p["seg"] == "c_head")
    top = hv[hv[:, 1] > np.percentile(hv[:, 1], 97)]
    head_top = [float(hv[:, 0].mean()), float(hv[:, 1].max()), float(top[:, 2].mean())]
    axis = BASIS @ PITCH_ROT @ np.array([-1.0, 0, 0])  # body axis, head to tail
    allv = np.vstack([p["v"] for p in parts])
    out = dict(
        source="NeuroMechFly v2 meshes from flygym 2.1.0 (Apache-2.0), micro-CT of an adult Drosophila",
        pitch=float(PITCH), thorax=thorax.tolist(), bodyAxis=axis.tolist(),
        head=to_fly(T["c_head"][:3, 3] + [0.25, 0, 0.1]).tolist(), headTop=head_top,
        bounds=[allv.min(0).tolist(), allv.max(0).tolist()],
        hero=hero, wings=wings, crowd=crowd, rig=rig,
    )
    (HERE / "fly_model.json").write_text(json.dumps(out, separators=(",", ":")))
    (HERE / "fly_model.bin").write_bytes(bytes(BLOB))
    tri = lambda d: d["index"]["length"] // 3
    print("hero tris:", {k: tri(v) for k, v in hero.items()}, "wing tris:", tri(wings["l"]), "crowd tris:", tri(crowd))
    print("bounds", np.round(allv.min(0), 3), np.round(allv.max(0), 3), "size", len(BLOB) // 1024, "KB")


if __name__ == "__main__":
    main()
