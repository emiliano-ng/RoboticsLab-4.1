from __future__ import annotations
import time
import numpy as np
import mujoco

from so101_control import (
    JointPID, PIDGains,
    get_q_qd_dict,
    apply_joint_torques_qfrc,
)

from typing import Protocol, Optional


class _PlotterProto(Protocol):
    def sample(self, m, d, now: float | None = None, q_des_current=None) -> None: ...


DEFAULT_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll"
]


# ============================================================
# 🔷 UTILS
# ============================================================

def lerp_pose(p0: dict[str, float], p1: dict[str, float], s: float):
    s = float(np.clip(s, 0.0, 1.0))
    out = {}
    for k in p0.keys():
        out[k] = (1.0 - s) * p0[k] + s * p1[k]
    return out


def build_default_pid(joint_names=DEFAULT_JOINTS) -> JointPID:

    gains = {
        "shoulder_pan":  PIDGains(kp=0.0, ki=0.0, kd=0.0, i_limit=2.0, tau_limit=8.0),
        "shoulder_lift": PIDGains(kp=0.0, ki=0.0, kd=0.0, i_limit=2.0, tau_limit=8.0),
        "elbow_flex":    PIDGains(kp=0.0, ki=0.0, kd=0.5, i_limit=2.0, tau_limit=8.0),
        "wrist_flex":    PIDGains(kp=0.0, ki=0.0, kd=0.5, i_limit=2.0, tau_limit=8.0),
        "wrist_roll":    PIDGains(kp=0.0, ki=0.0, kd=0.8, i_limit=2.0, tau_limit=8.0),
    }

    for jn in joint_names:
        if jn not in gains:
            gains[jn] = PIDGains(
                kp=25.0,
                ki=0.3,
                kd=1.0,
                i_limit=2.0,
                tau_limit=6.0
            )

    return JointPID(joint_names, gains)


# ============================================================
# 🔷 STEP SIM
# ============================================================

def step_sim(m, d, viewer, realtime: bool,
             plotter: Optional[_PlotterProto] = None):

    mujoco.mj_step(m, d)

    if plotter is not None:
        plotter.sample(m, d)

    if viewer is not None:
        viewer.sync()

    if realtime:
        time.sleep(m.opt.timestep)


# ============================================================
# 🔷 MOVE TO POSE (SIN PERTURBACIONES)
# ============================================================

def move_to_pose_pid(
    m, d, viewer,
    target_pose_deg,
    duration=2.0,
    realtime=True,
    joint_names=DEFAULT_JOINTS,
    pid=None,
    perturb=None,  # <- ya no se usa
    plotter=None,
):

    if pid is None:
        pid = build_default_pid(joint_names)

    pid.reset()

    q0, _ = get_q_qd_dict(m, d, joint_names)
    qT = {jn: np.deg2rad(target_pose_deg[jn]) for jn in joint_names}

    steps = int(max(1, duration / m.opt.timestep))
    t0 = float(d.time)

    for _ in range(steps):

        t = float(d.time)
        s = (t - t0) / max(duration, 1e-9)
        q_des = lerp_pose(q0, qT, s)

        q, qd = get_q_qd_dict(m, d, joint_names)

        tau_pid = pid.compute(q, qd, q_des, m.opt.timestep)

        apply_joint_torques_qfrc(m, d, joint_names, tau_pid)

        if plotter is not None:
            plotter.sample(m, d, q_des_current=q_des)

        step_sim(m, d, viewer, realtime=realtime)


# ============================================================
# 🔷 HOLD POSITION (SIN PERTURBACIONES)
# ============================================================

def hold_position_pid(
    m, d, viewer,
    hold_pose_deg,
    duration=2.0,
    realtime=True,
    joint_names=DEFAULT_JOINTS,
    pid=None,
    perturb=None,  # <- ya no se usa
    plotter=None,
):

    if pid is None:
        pid = build_default_pid(joint_names)

    pid.reset()

    q_des = {jn: np.deg2rad(hold_pose_deg[jn]) for jn in joint_names}
    steps = int(max(1, duration / m.opt.timestep))

    for _ in range(steps):

        q, qd = get_q_qd_dict(m, d, joint_names)

        tau_pid = pid.compute(q, qd, q_des, m.opt.timestep)
        if abs(tau_pid["shoulder_lift"]) > 17:
            print("SATURANDO shoulder_lift:", tau_pid["shoulder_lift"])

        apply_joint_torques_qfrc(m, d, joint_names, tau_pid)

        if plotter is not None:
            plotter.sample(m, d, q_des_current=q_des)

        step_sim(m, d, viewer, realtime=realtime)
