import argparse
import csv
import math
import os
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_URDF_CANDIDATES = [
    os.path.join(PROJECT_DIR, "..", "..", "hardware", "mechanical", "urdf", "panda_arm_hand.urdf"),
    os.path.join(os.path.dirname(os.__file__), "site-packages", "isaacsim", "exts", "isaacsim.asset.importer.urdf",
                 "data", "urdf", "robots", "franka_description", "robots", "panda_arm_hand.urdf"),
    r"C:\Users\uSeR\env_isaaclab\Lib\site-packages\isaacsim\exts\isaacsim.asset.importer.urdf"
    r"\data\urdf\robots\franka_description\robots\panda_arm_hand.urdf",
]
DEFAULT_URDF = next((os.path.normpath(p) for p in _URDF_CANDIDATES if os.path.isfile(p)), _URDF_CANDIDATES[-1])
OUT = os.path.join(PROJECT_DIR, "report", "urdf")
HOME = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
TCP_OFFSET = 0.1034
SPAWN_R = (0.40, 0.70)
SPAWN_DEG = 160.0


def vec(s, n=3):
    return np.array([float(x) for x in s.split()]) if s else np.zeros(n)


def parse_urdf(path):
    root = ET.parse(path).getroot()
    links, joints = {}, []
    for l in root.findall("link"):
        d = {"name": l.get("name"), "mass": None, "inertia": None, "com": np.zeros(3), "visual": "", "collision": ""}
        inert = l.find("inertial")
        if inert is not None:
            d["mass"] = float(inert.find("mass").get("value"))
            i = inert.find("inertia")
            d["inertia"] = [float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            o = inert.find("origin")
            if o is not None:
                d["com"] = vec(o.get("xyz"))
        for tag in ("visual", "collision"):
            m = l.find(f"{tag}/geometry/mesh")
            if m is not None:
                d[tag] = os.path.basename(m.get("filename"))
        links[d["name"]] = d
    for j in root.findall("joint"):
        o = j.find("origin")
        lim = j.find("limit")
        ax = j.find("axis")
        mimic = j.find("mimic")
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": vec(o.get("xyz")) if o is not None else np.zeros(3),
            "rpy": vec(o.get("rpy")) if o is not None else np.zeros(3),
            "axis": vec(ax.get("xyz")) if ax is not None else np.array([1.0, 0, 0]),
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            "velocity": float(lim.get("velocity")) if lim is not None and lim.get("velocity") else None,
            "effort": float(lim.get("effort")) if lim is not None and lim.get("effort") else None,
            "mimic": mimic.get("joint") if mimic is not None else "",
        })
    return links, joints


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rot_axis(axis, q):
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * k + (1 - math.cos(q)) * k @ k


def tf(R=np.eye(3), p=np.zeros(3)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def fk_urdf(joints, q, tip="panda_hand"):
    by_child = {j["child"]: j for j in joints}
    chain = []
    link = tip
    while link in by_child:
        j = by_child[link]
        chain.append(j)
        link = j["parent"]
    chain.reverse()
    T = np.eye(4)
    poses, frames = {chain[0]["parent"]: T.copy()}, []
    qi = 0
    for j in chain:
        T = T @ tf(rot_rpy(*j["rpy"]), j["xyz"])
        if j["type"] in ("revolute", "continuous"):
            frames.append((T.copy(), j["axis"], j["name"]))
            T = T @ tf(rot_axis(j["axis"], q[qi]))
            qi += 1
        poses[j["child"]] = T.copy()
    return poses, frames


def dh_from_urdf(joints):
    rows = []
    for j in joints:
        if j["type"] != "revolute":
            continue
        assert abs(j["rpy"][1]) < 1e-9 and abs(j["rpy"][2]) < 1e-9, "only roll-type origins are handled"
        R = rot_rpy(*j["rpy"])
        p = R.T @ j["xyz"]
        assert abs(p[1]) < 1e-9, "origin is not DH-expressible"
        rows.append({"joint": j["name"], "a": j["xyz"][0], "alpha": j["rpy"][0], "d": p[2]})
    return rows


def fk_dh(rows, q, flange=0.107, hand_yaw=-math.pi / 4):
    T = np.eye(4)
    for r, qi in zip(rows, q):
        ca, sa = math.cos(r["alpha"]), math.sin(r["alpha"])
        Tx = np.array([[1, 0, 0, r["a"]], [0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]])
        ct, st = math.cos(qi), math.sin(qi)
        Tz = np.array([[ct, -st, 0, 0], [st, ct, 0, 0], [0, 0, 1, r["d"]], [0, 0, 0, 1]])
        T = T @ Tx @ Tz
    return T @ tf(np.eye(3), np.array([0, 0, flange])) @ tf(rot_rpy(0, 0, hand_yaw))


def write_tables(links, joints):
    deg = math.degrees
    with open(os.path.join(OUT, "joint_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["joint", "type", "parent", "child", "origin_xyz_m", "origin_rpy_rad", "axis",
                    "lower_rad", "upper_rad", "lower_deg", "upper_deg", "velocity", "effort", "mimic"])
        for j in joints:
            w.writerow([j["name"], j["type"], j["parent"], j["child"], " ".join(f"{v:g}" for v in j["xyz"]),
                        " ".join(f"{v:.4f}" for v in j["rpy"]), " ".join(f"{v:g}" for v in j["axis"]),
                        j["lower"], j["upper"],
                        None if j["lower"] is None else round(deg(j["lower"]), 1),
                        None if j["upper"] is None else round(deg(j["upper"]), 1),
                        j["velocity"], j["effort"], j["mimic"]])
    lines = [
        "| Joint | Type | Parent → Child | Origin xyz (m) | Origin rpy (°) | Axis | Range | Max velocity | Max effort |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for j in joints:
        if j["type"] == "revolute":
            rng = f"{deg(j['lower']):.0f}° … {deg(j['upper']):.0f}°"
            vel = f"{j['velocity']:.3f} rad/s ({deg(j['velocity']):.0f}°/s)"
            eff = f"{j['effort']:g} Nm"
        elif j["type"] == "prismatic":
            rng = f"{j['lower'] * 100:g} … {j['upper'] * 100:g} cm"
            vel = f"{j['velocity']:g} m/s"
            eff = f"{j['effort']:g} N"
        else:
            rng = vel = eff = "—"
        rpy = " ".join(f"{deg(v):.0f}" for v in j["rpy"])
        xyz = " ".join(f"{v:g}" for v in j["xyz"])
        ax = " ".join(f"{v:g}" for v in j["axis"]) if j["type"] != "fixed" else "—"
        typ = j["type"] + (f" (mimic {j['mimic']})" if j["mimic"] else "")
        lines.append(f"| `{j['name']}` | {typ} | {j['parent']} → {j['child']} | {xyz} | {rpy} | {ax} | {rng} | {vel} | {eff} |")
    open(os.path.join(OUT, "joint_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    with open(os.path.join(OUT, "link_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["link", "mass_kg", "ixx", "iyy", "izz", "ixy", "ixz", "iyz", "visual_mesh", "collision_mesh"])
        for l in links.values():
            w.writerow([l["name"], l["mass"], *(l["inertia"] or [None] * 6), l["visual"], l["collision"]])
    total = sum(l["mass"] or 0 for l in links.values())
    lines = ["| Link | Mass (kg) | Ixx | Iyy | Izz (kg·m²) | Visual mesh | Collision mesh |", "|---|---|---|---|---|---|---|"]
    for l in links.values():
        if l["mass"] is None:
            lines.append(f"| `{l['name']}` | — (frame only) | | | | | |")
            continue
        ixx, iyy, izz = l["inertia"][:3]
        lines.append(f"| `{l['name']}` | {l['mass']:.3f} | {ixx:.2e} | {iyy:.2e} | {izz:.2e} | {l['visual']} | {l['collision']} |")
    lines.append(f"| **total** | **{total:.2f}** | | | | | |")
    open(os.path.join(OUT, "link_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return total


def write_dh(rows):
    lines = [
        "Modified DH (Craig) parameters of the Panda arm, derived from the URDF joint origins.",
        "T_i = RotX(alpha_{i-1}) · TransX(a_{i-1}) · RotZ(theta_i) · TransZ(d_i)",
        "",
        "| i | a_{i-1} (m) | d_i (m) | alpha_{i-1} (°) | theta_i |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['a']:g} | {r['d']:g} | {math.degrees(r['alpha']):.0f} | q{i} |")
    lines.append("| flange (`panda_joint8`, fixed) | 0 | 0.107 | 0 | 0 |")
    lines.append("| hand (`panda_hand_joint`, fixed) | 0 | 0 | 0 | −45° (yaw) |")
    lines.append(f"| TCP (between fingertips) | 0 | {TCP_OFFSET} | 0 | 0 |")
    open(os.path.join(OUT, "dh_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def fk_check(joints, rows, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    rev = [j for j in joints if j["type"] == "revolute"]
    lo, hi = np.array([j["lower"] for j in rev]), np.array([j["upper"] for j in rev])
    errs_p, errs_r = [], []
    for _ in range(n):
        q = rng.uniform(lo, hi)
        A = fk_urdf(joints, q)[0]["panda_hand"]
        B = fk_dh(rows, q)
        errs_p.append(np.linalg.norm(A[:3, 3] - B[:3, 3]))
        errs_r.append(np.linalg.norm(A[:3, :3] - B[:3, :3]))
    home = fk_urdf(joints, HOME)[0]["panda_hand"]
    tcp = home @ tf(np.eye(3), np.array([0, 0, TCP_OFFSET]))
    txt = [
        "Forward kinematics check: URDF chain (origin xyz/rpy + axis) vs modified DH table",
        f"random joint configs inside the URDF limits : {n}",
        f"max position difference of panda_hand      : {max(errs_p):.2e} m",
        f"max rotation-matrix difference (Frobenius) : {max(errs_r):.2e}",
        "",
        f"home pose q = {np.round(HOME, 3).tolist()} rad",
        f"panda_hand position at home : {np.round(home[:3, 3], 4).tolist()} m (robot base frame)",
        f"TCP position at home        : {np.round(tcp[:3, 3], 4).tolist()} m",
    ]
    open(os.path.join(OUT, "fk_check.txt"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))
    return max(errs_p)


def plot_tree(joints):
    color = {"revolute": "#1f77b4", "prismatic": "#d62728", "fixed": "#7f7f7f"}
    names = ["panda_link0"] + [j["child"] for j in joints if j["child"] not in ("panda_leftfinger", "panda_rightfinger")]
    fig, ax = plt.subplots(figsize=(7.5, 11))
    ax.axis("off")
    y = {n: -i for i, n in enumerate(names)}
    for n in names:
        ax.text(0, y[n], n, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff6d5" if n != "panda_hand" else "#e2f0d9", ec="k"))
    fy = y["panda_hand"] - 1.3
    for k, n in enumerate(("panda_leftfinger", "panda_rightfinger")):
        y[n] = fy
        ax.text(-1.2 + 2.4 * k, fy, n, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="#fde2e2", ec="k"))
    xs = {"panda_leftfinger": -1.2, "panda_rightfinger": 1.2}
    for j in joints:
        x1, y1 = 0, y[j["parent"]]
        x2, y2 = xs.get(j["child"], 0), y[j["child"]]
        ax.annotate("", xy=(x2, y2 + 0.28), xytext=(x1, y1 - 0.28),
                    arrowprops=dict(arrowstyle="-|>", color=color[j["type"]], lw=2))
        if j["type"] == "revolute":
            lab = f"{j['name']}  revolute  {math.degrees(j['lower']):.0f}°…{math.degrees(j['upper']):.0f}°  {j['effort']:g} Nm"
        elif j["type"] == "prismatic":
            lab = f"{j['name']}  prismatic 0…{j['upper'] * 100:g} cm" + ("  (mimic)" if j["mimic"] else "")
        else:
            lab = f"{j['name']}  fixed  z+{j['xyz'][2]:g} m" if j["xyz"][2] else f"{j['name']}  fixed  yaw −45°"
        lx = (x1 + x2) / 2 + (0.15 if x2 >= 0 else -0.15)
        ax.text(lx + (0.1 if x2 == 0 else 0), (y1 + y2) / 2, lab, fontsize=8, color=color[j["type"]],
                ha="left" if x2 >= 0 else "right", va="center")
    for k, (t, c) in enumerate(color.items()):
        ax.text(-2.4, 0.2 - 0.45 * k, f"— {t}", color=c, fontsize=10, fontweight="bold")
    ax.set_xlim(-2.6, 3.4)
    ax.set_ylim(fy - 0.8, 0.8)
    ax.set_title("Franka Panda URDF — kinematic tree (links = boxes, joints = arrows)", fontsize=11)
    fig.savefig(os.path.join(OUT, "kinematic_tree.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_frames_3d(joints):
    poses, frames = fk_urdf(joints, HOME)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    order = ["panda_link0"] + [f"panda_link{i}" for i in range(1, 9)] + ["panda_hand"]
    pts = np.array([poses[n][:3, 3] for n in order])
    tcp = (poses["panda_hand"] @ tf(np.eye(3), np.array([0, 0, TCP_OFFSET])))[:3, 3]
    pts = np.vstack([pts, tcp])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="0.3", lw=4, ms=5, label="link chain (frame origins)")
    for k, (T, axis, name) in enumerate(frames):
        p = T[:3, 3]
        a = T[:3, :3] @ axis * 0.12
        ax.quiver(*p, *a, color="C3", lw=2.5, arrow_length_ratio=0.25)
        ax.text(*(p + a * 1.15), f"J{k + 1}", color="C3", fontsize=11, fontweight="bold")
    ax.scatter(*tcp, color="C2", s=80, label="TCP (between fingertips)")
    ax.quiver(0, 0, 0, 0.15, 0, 0, color="r")
    ax.quiver(0, 0, 0, 0, 0.15, 0, color="g")
    ax.quiver(0, 0, 0, 0, 0, 0.15, color="b")
    ax.text(0.17, 0, 0, "x (front)", color="r")
    ax.set_xlim(-0.4, 0.6)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Joint frames from the URDF at the home pose (red arrows = rotation axes J1–J7)", fontsize=10)
    fig.savefig(os.path.join(OUT, "robot_frames_3d.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_limits(joints):
    rev = [j for j in joints if j["type"] == "revolute"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, j in enumerate(rev):
        lo, hi = math.degrees(j["lower"]), math.degrees(j["upper"])
        ax.barh(k, hi - lo, left=lo, color="#9ecae1", edgecolor="k")
        ax.plot(math.degrees(HOME[k]), k, "kD", ms=6)
        ax.text(hi + 4, k, f"{lo:.0f}…{hi:.0f}°", va="center", fontsize=8)
    ax.plot([], [], "kD", label="home pose of the task")
    ax.set_yticks(range(len(rev)))
    ax.set_yticklabels([f"J{k + 1}" for k in range(len(rev))])
    ax.invert_yaxis()
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlim(-200, 260)
    ax.set_xlabel("joint angle (deg)")
    ax.set_title("Joint ranges from the URDF <limit lower/upper>")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(OUT, "joint_limits.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_workspace(joints, rows, n=40000, seed=1):
    rng = np.random.default_rng(seed)
    rev = [j for j in joints if j["type"] == "revolute"]
    lo, hi = np.array([j["lower"] for j in rev]), np.array([j["upper"] for j in rev])
    P = np.empty((n, 3))
    for i in range(n):
        T = fk_dh(rows, rng.uniform(lo, hi)) @ tf(np.eye(3), np.array([0, 0, TCP_OFFSET]))
        P[i] = T[:3, 3]
    r = np.hypot(P[:, 0], P[:, 1])
    reach = np.linalg.norm(P - np.array([0, 0, 0.333]), axis=1).max()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5))
    near_table = np.abs(P[:, 2] - 0.05) < 0.05
    a1.scatter(P[:, 0], P[:, 1], s=0.3, c="0.75", label="all reachable TCP positions")
    a1.scatter(P[near_table, 0], P[near_table, 1], s=0.6, c="C0", label="TCP at cube height (z = 0–10 cm)")
    th = np.linspace(-math.radians(SPAWN_DEG), math.radians(SPAWN_DEG), 200)
    for rr in SPAWN_R:
        a1.plot(rr * np.cos(th), rr * np.sin(th), "C3", lw=2)
    a1.plot([], [], "C3", lw=2, label=f"cube spawn area: r {SPAWN_R[0]}–{SPAWN_R[1]} m, ±{SPAWN_DEG:.0f}°")
    lim1 = math.radians(math.degrees(rev[0]["upper"]))
    for s in (1, -1):
        a1.plot([0, 0.9 * math.cos(s * lim1)], [0, 0.9 * math.sin(s * lim1)], "k--", lw=1)
    a1.text(-0.85, 0.05, "J1 limit ±166°\n(dashed)", fontsize=8)
    a1.set_aspect("equal")
    a1.set_xlabel("x (m, front)")
    a1.set_ylabel("y (m)")
    a1.set_title("Top view")
    a1.legend(loc="lower left", fontsize=7, markerscale=10)
    a2.scatter(r, P[:, 2], s=0.3, c="0.75")
    a2.axvspan(*SPAWN_R, color="C3", alpha=0.15, label="cube spawn radius")
    a2.axhline(0.0, color="k", lw=1)
    a2.text(0.02, 0.01, "table top", fontsize=8)
    a2.set_xlabel("horizontal distance from J1 axis (m)")
    a2.set_ylabel("z (m)")
    a2.set_title(f"Side view (max reach from the shoulder ≈ {reach:.3f} m)")
    a2.legend(loc="upper right", fontsize=8)
    a2.set_aspect("equal")
    fig.suptitle(f"Workspace of the TCP from {n} random joint configs inside the URDF limits")
    fig.savefig(os.path.join(OUT, "workspace.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return reach, r.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    links, joints = parse_urdf(args.urdf)
    print(f"{len(links)} links, {len(joints)} joints "
          f"({sum(j['type'] == 'revolute' for j in joints)} revolute, "
          f"{sum(j['type'] == 'prismatic' for j in joints)} prismatic, {sum(j['type'] == 'fixed' for j in joints)} fixed)")
    total = write_tables(links, joints)
    print(f"total mass {total:.2f} kg")
    rows = dh_from_urdf(joints)
    write_dh(rows)
    fk_check(joints, rows)
    plot_tree(joints)
    plot_frames_3d(joints)
    plot_limits(joints)
    reach, rmax = plot_workspace(joints, rows)
    print(f"max reach from shoulder {reach:.3f} m, max horizontal TCP radius {rmax:.3f} m")
    print(f"outputs in {OUT}")


if __name__ == "__main__":
    main()
