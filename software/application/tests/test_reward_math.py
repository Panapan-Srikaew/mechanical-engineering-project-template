import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(
    HERE, "..", "source", "franka_pick_place", "franka_pick_place", "tasks", "pick_place", "mdp", "reward_math.py"
)
spec = importlib.util.spec_from_file_location("reward_math", PATH)
rm = importlib.util.module_from_spec(spec)
sys.modules["reward_math"] = rm
spec.loader.exec_module(rm)

P = rm.PickPlaceParams()
W = rm.DEFAULT_WEIGHTS
REST = P.object_rest_z


def sig(d_ee, d_goal, obj_z=REST, goal_z=REST, opening=0.08, speed=0.0, was_lifted=None):
    a = lambda v: np.array([v], dtype=np.float64)
    wl = None if was_lifted is None else np.array([was_lifted])
    s = rm.stage_signals(a(d_ee), a(d_goal), a(obj_z), a(goal_z), a(opening), a(speed), P, was_lifted=wl)
    return {k: float(v[0]) for k, v in s.items()}


def total(s):
    return float(rm.weighted_step_reward(s, W))


passed = 0


def check(name, cond):
    global passed
    print(("PASS  " if cond else "FAIL  ") + name)
    assert cond, name
    passed += 1


check("reach decreases with distance", sig(0.02, 0.4)["reach"] > sig(0.3, 0.4)["reach"] > sig(0.8, 0.4)["reach"])

check("open gripper near cube is NOT grasp", sig(0.01, 0.4, opening=0.08)["grasp"] == 0)
check("empty closed gripper is NOT grasp", sig(0.01, 0.4, opening=0.0)["grasp"] == 0)
check("fingers on cube (4.1cm) IS grasp", sig(0.01, 0.4, opening=0.041)["grasp"] == 1)
check("closed on cube but far away is NOT grasp", sig(0.2, 0.4, opening=0.041)["grasp"] == 0)

check("lift needs grasp", sig(0.01, 0.4, obj_z=0.15, opening=0.08)["lift"] == 0)
check("lift when grasped and raised", sig(0.01, 0.3, obj_z=0.15, opening=0.041)["lift"] == 1)
check("no lift when grasped but on table", sig(0.01, 0.4, obj_z=REST, opening=0.041)["lift"] == 0)

a = sig(0.01, 0.40, obj_z=0.15, opening=0.041)
b = sig(0.01, 0.10, obj_z=0.15, opening=0.041)
check("transport grows as object nears goal", b["transport_coarse"] > a["transport_coarse"] and b["transport_fine"] > a["transport_fine"])
check("pushing (no grasp) far from goal gives no transport", sig(0.2, 0.40, opening=0.08)["transport_coarse"] == 0)
check("dragging on the table (grasped, not lifted) gives no transport", sig(0.01, 0.2, opening=0.041)["transport_coarse"] == 0)
check("lift is dense: higher is better up to lift_height",
      sig(0.01, 0.4, obj_z=REST + 0.01, opening=0.041)["lift"] < sig(0.01, 0.4, obj_z=REST + 0.02, opening=0.041)["lift"])

holding_at_goal = sig(0.01, 0.005, opening=0.041)
releasing = sig(0.03, 0.005, opening=0.075)
retreated = sig(0.08, 0.005, opening=0.08)
check("holding at table goal: no success yet", holding_at_goal["success"] == 0)
check("opened but still touching: no success yet", releasing["success"] == 0)
check("opened + retreated + settled => SUCCESS", retreated["success"] == 1)
check("per-step reward: released > holding at goal  (no 'hold forever' exploit)", total(retreated) > total(holding_at_goal))
check("per-step reward: releasing > holding at goal", total(releasing) > total(holding_at_goal))
check("moving cube at goal is not success", sig(0.08, 0.005, opening=0.08, speed=0.5)["success"] == 0)
check("reach term vanishes once cube is at table goal (lets hand retreat)", retreated["reach"] == 0)

far_idle = total(sig(0.5, 0.5))
reaching = total(sig(0.05, 0.5))
grasping = total(sig(0.01, 0.5, opening=0.041))
carrying = total(sig(0.01, 0.2, obj_z=0.15, opening=0.041))
check("reward ordering idle < reach < grasp < carry < released", far_idle < reaching < grasping < carrying < total(retreated))

air = 0.20
at_air_goal = sig(0.01, 0.005, obj_z=air, goal_z=air, opening=0.041, speed=0.0)
check("elevated goal: success = hold at goal (no release needed)", at_air_goal["success"] == 1)
check("elevated goal: released cube (falling) is not success", sig(0.1, 0.15, obj_z=0.05, goal_z=air, opening=0.08)["success"] == 0)

pushed = sig(0.08, 0.005, opening=0.08, was_lifted=False)
check("pushed onto target (never lifted) is NOT success", pushed["success"] == 0)
check("pushed onto target gets no place / transport reward", pushed["place"] == 0 and pushed["transport_fine"] == 0)
check("same state after a real pick IS success", sig(0.08, 0.005, opening=0.08, was_lifted=True)["success"] == 1)
check("grasp + lift sets the latch", sig(0.01, 0.3, obj_z=0.15, opening=0.041, was_lifted=False)["lifted_latch"] == 1)
check("latch stays off while only reaching", sig(0.01, 0.3, opening=0.08, was_lifted=False)["lifted_latch"] == 0)
check("reach still has a gradient 0.4 m away", sig(0.4, 0.5)["reach"] > sig(0.6, 0.5)["reach"] + 0.02)

def sig_cmd(close, **kw):
    a = lambda v: np.array([v], dtype=np.float64)
    s = rm.stage_signals(a(0.01), a(0.3), a(kw.get("obj_z", 0.15)), a(REST), a(0.041), a(0.0), P,
                         close_cmd=np.array([close]))
    return {k: float(v[0]) for k, v in s.items()}


check("fingers at cube width but gripper commanded OPEN is NOT grasp/lift", sig_cmd(False)["grasp"] == 0 and sig_cmd(False)["lift"] == 0)
check("gripper commanded CLOSE at cube width IS grasp+lift", sig_cmd(True)["grasp"] == 1 and sig_cmd(True)["lift"] == 1)

P360 = rm.PickPlaceParams(place_tol=0.015, fine_std=0.05)


def sig360(d_goal):
    one = lambda v: np.array([v], dtype=float)
    return rm.stage_signals(one(0.08), one(d_goal), one(REST), one(REST), one(0.08), one(0.0), P360,
                            was_lifted=np.array([True]))


check("360: released 2.5 cm from target is NOT success (tolerance 1.5 cm)", sig360(0.025)["success"][0] == 0)
check("360: released 1 cm from target IS success", sig360(0.010)["success"][0] == 1)
check("360: tolerance 3 cm (curriculum start) accepts 2.5 cm",
      rm.stage_signals(*(np.array([v]) for v in (0.08, 0.025, REST, REST, 0.08, 0.0)),
                       rm.PickPlaceParams(place_tol=0.03, fine_std=0.05), was_lifted=np.array([True]))["success"][0] == 1)

PV = rm.PickPlaceParams(place_tol=0.015, fine_std=0.05, lift_hold_after_pick=True)
W360 = dict(W, transport_fine=15.0)


def s360(d_ee, d_goal, d_xy, obj_z, opening, close):
    one = lambda v: np.array([v], dtype=float)
    return rm.stage_signals(one(d_ee), one(d_goal), one(obj_z), one(REST), one(opening), one(0.0), PV,
                            was_lifted=np.array([True]), close_cmd=np.array([close]), d_goal_xy=one(d_xy))


hover = s360(0.01, 0.06, 0.0, REST + 0.06, 0.041, True)
lowered = s360(0.01, 0.005, 0.0, REST + 0.005, 0.041, True)
released = s360(0.08, 0.005, 0.0, REST + 0.005, 0.08, False)
r = lambda s: rm.weighted_step_reward(s, W360)[0]
check("360: lowering onto the goal pays more than hovering above it", r(lowered) > r(hover))
check("360: releasing at the goal pays more than holding it there", r(released) > r(lowered))
check("360: lift reward kept when the held cube is lowered", lowered["lift"][0] == 1)
high = s360(0.01, 0.25, 0.0, REST + 0.25, 0.041, True)
check("360: holding the cube high above the goal pays less than hovering low", r(high) < r(hover))

rng = np.random.default_rng(0)
N = 5000
args = (
    rng.uniform(0, 1, N), rng.uniform(0, 1, N), rng.uniform(0, 0.4, N), rng.uniform(REST, 0.3, N),
    rng.uniform(0, 0.08, N), rng.uniform(0, 1, N),
)
s = rm.stage_signals(*args, P)
ok = all(np.all((v >= 0) & (v <= 1.0 + 1e-9)) and np.all(np.isfinite(v)) for v in s.values())
check("all signals finite and within [0, 1] on random inputs", ok)

print(f"\nAll {passed} checks passed.")
