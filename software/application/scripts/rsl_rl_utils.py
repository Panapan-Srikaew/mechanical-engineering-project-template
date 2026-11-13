import copy
import glob
import os
import re


def adapt_agent_cfg(agent_cfg):
    from importlib import metadata

    from packaging import version as pkg_version

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

    installed = metadata.version("rsl-rl-lib")
    last_err = None
    for arg in (installed, pkg_version.parse(installed)):
        try:
            out = handle_deprecated_rsl_rl_cfg(copy.deepcopy(agent_cfg), arg)
            print(f"[INFO] rsl_rl {installed}: agent cfg adapted")
            return out
        except Exception as e:
            last_err = e
    raise last_err


def newest_checkpoint(root: str) -> str:
    files = glob.glob(os.path.join(root, "*", "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"no checkpoints under {root}")
    newest_run = os.path.dirname(max(files, key=os.path.getmtime))
    run_files = [f for f in files if os.path.dirname(f) == newest_run]
    return max(run_files, key=lambda f: int(re.findall(r"model_(\d+)\.pt", f)[0]))


def add_legacy_args(parser):
    g = parser.add_argument_group("replay older runs (settings per run: RESULTS.md)")
    g.add_argument("--arm_scale", type=float, default=None, help="IK arm action scale (runs 8-9: 0.5)")
    g.add_argument("--action_clip", type=float, default=None, help="in-env IK action clip (runs 8-9: 1000 = none)")
    g.add_argument("--no_object_to_target", action="store_true", help="drop the object_to_target obs (runs 1-7)")
    g.add_argument("--clip_actions", type=float, default=None, help="clip in the rsl_rl wrapper (runs 10-11: 1.0)")
    g.add_argument("--no_arm_state", action="store_true", help="drop the arm_state obs (runs 13-14)")
    g.add_argument("--nullspace_gain", type=float, default=None, help="IK null-space posture gain (run 18+: 0.05, older: 0)")
    v = parser.add_argument_group("evaluation variations (robot model, robustness, seed)")
    v.add_argument("--env_seed", type=int, default=None, help="seed of the environment (spawn positions, randomization)")
    v.add_argument("--robot_urdf", type=str, default=None,
                   help="build the robot directly from this URDF (Isaac Sim URDF importer) instead of franka.usd")
    v.add_argument("--urdf_collision", type=str, default="Convex Hull",
                   help="URDF importer collision approximation: 'Convex Hull' | 'Convex Decomposition' | ...")
    v.add_argument("--urdf_finger_vel", type=float, default=None,
                   help="finger joint velocity limit in m/s for the URDF robot (URDF: 0.2; the importer writes 11.46 = 0.2 rad->deg)")
    v.add_argument("--inertia_scale", type=float, default=None,
                   help="scale every robot link inertia tensor at startup (franka.usd = URDF x 1e-4)")
    v.add_argument("--urdf_raw", action="store_true",
                   help="use the importer output as it is (without writing the robot's physx settings on every link)")
    v.add_argument("--urdf_mimic_to_normal", action="store_true",
                   help="import the <mimic> finger joint as a normal driven joint (as in franka.usd)")
    v.add_argument("--mass_range", type=str, default=None, help="cube mass scale range 'lo,hi' (training: 0.7,1.5)")
    v.add_argument("--friction_scale", type=float, default=None, help="scale the cube + finger friction ranges")
    v.add_argument("--obs_noise", action="store_true", help="keep the training observation noise on during eval")
    v.add_argument("--obs_noise_scale", type=float, default=None, help="multiply the observation noise (implies --obs_noise)")


def _franka_package_dir(urdf_path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(urdf_path)))
    if os.path.isdir(os.path.join(here, "meshes")):
        return here
    import isaacsim

    base = os.path.dirname(isaacsim.__file__)
    return os.path.join(base, "exts", "isaacsim.asset.importer.urdf", "data", "urdf", "robots", "franka_description")


def _author_physx_settings(usd_path, rigid_props, articulation_props):
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    n_body = n_art = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
            if rigid_props.disable_gravity is not None:
                prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(bool(rigid_props.disable_gravity))
            if rigid_props.max_depenetration_velocity is not None:
                prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(
                    float(rigid_props.max_depenetration_velocity))
            n_body += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.AddAppliedSchema("PhysxArticulationAPI")
            a = articulation_props
            if a.enabled_self_collisions is not None:
                prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(bool(a.enabled_self_collisions))
            if a.solver_position_iteration_count is not None:
                prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
                    int(a.solver_position_iteration_count))
            if a.solver_velocity_iteration_count is not None:
                prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(
                    int(a.solver_velocity_iteration_count))
            n_art += 1
    stage.GetRootLayer().Save()
    print(f"[INFO] physx settings written on {n_body} links / {n_art} articulation root(s) of {usd_path}")


def apply_robot_urdf(env_cfg, urdf_path, collision_type="Convex Hull", mimic_to_normal=False, raw=False):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import UrdfConverter

    robot = env_cfg.scene.robot
    old = robot.spawn
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conv_cfg = sim_utils.UrdfConverterCfg(
        asset_path=os.path.abspath(urdf_path),
        usd_dir=os.path.join(project, "report", "urdf_usd", "raw" if raw else "matched"),
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=True,
        collision_type=collision_type,
        convert_mimic_joints_to_normal_joints=mimic_to_normal,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        ros_package_paths=[{"name": "franka_description", "path": _franka_package_dir(urdf_path)}],
        force_usd_conversion=True,
    )
    usd_path = UrdfConverter(conv_cfg).usd_path
    if not raw:
        _author_physx_settings(usd_path, old.rigid_props, old.articulation_props)
    robot.spawn = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=old.rigid_props,
        articulation_props=old.articulation_props,
        activate_contact_sensors=old.activate_contact_sensors,
    )
    chain = [f"panda_link{i}" for i in range(9)] + ["panda_hand"]
    nested = {name: "Geometry/" + "/".join(chain[: k + 1]) for k, name in enumerate(chain)}

    def fix(path):
        for name, sub in nested.items():
            if path.endswith(f"/Robot/{name}"):
                return path[: -len(name)] + sub
        return path

    ee = getattr(env_cfg.scene, "ee_frame", None)
    if ee is not None:
        ee.prim_path = fix(ee.prim_path)
        for frame in ee.target_frames:
            frame.prim_path = fix(frame.prim_path)
    print(f"[INFO] robot built from URDF: {urdf_path} (collision: {collision_type}, mimic -> normal: {mimic_to_normal}, raw: {raw})")


def apply_eval_variations(env_cfg, args):
    if args.env_seed is not None:
        env_cfg.seed = args.env_seed
    if args.robot_urdf:
        apply_robot_urdf(env_cfg, args.robot_urdf, args.urdf_collision, args.urdf_mimic_to_normal, raw=args.urdf_raw)
        if args.urdf_finger_vel is not None:
            env_cfg.scene.robot.actuators["panda_hand"].velocity_limit_sim = args.urdf_finger_vel
            print(f"[INFO] finger velocity limit set to {args.urdf_finger_vel} m/s")
    ev = env_cfg.events
    if args.inertia_scale is not None:
        from isaaclab.envs import mdp as base_mdp
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        ev.robot_inertia_scale = EventTermCfg(
            func=base_mdp.randomize_rigid_body_inertia,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "inertia_distribution_params": (args.inertia_scale, args.inertia_scale),
                    "operation": "scale", "diagonal_only": False},
        )
        print(f"[INFO] robot link inertia scaled by {args.inertia_scale:g}")
    if args.mass_range and getattr(ev, "object_mass", None) is not None:
        ev.object_mass.params["mass_distribution_params"] = tuple(float(x) for x in args.mass_range.split(","))
    if args.friction_scale is not None:
        for name in ("object_material", "finger_material"):
            term = getattr(ev, name, None)
            if term is not None:
                for key in ("static_friction_range", "dynamic_friction_range"):
                    lo, hi = term.params[key]
                    term.params[key] = (lo * args.friction_scale, hi * args.friction_scale)
    if args.obs_noise or args.obs_noise_scale is not None:
        obs = env_cfg.observations.policy
        obs.enable_corruption = True
        if args.obs_noise_scale is not None:
            for term in obs.__dict__.values():
                noise = getattr(term, "noise", None)
                if noise is not None and hasattr(noise, "n_min"):
                    noise.n_min *= args.obs_noise_scale
                    noise.n_max *= args.obs_noise_scale


def apply_legacy_overrides(env_cfg, args):
    apply_eval_variations(env_cfg, args)
    arm = env_cfg.actions.arm_action
    if args.arm_scale is not None:
        arm.scale = args.arm_scale
    if args.action_clip is not None:
        arm.action_clip = args.action_clip
    if args.no_object_to_target:
        env_cfg.observations.policy.object_to_target = None
    if args.no_arm_state:
        env_cfg.observations.policy.arm_state = None
    if args.nullspace_gain is not None:
        arm.nullspace_gain = args.nullspace_gain
