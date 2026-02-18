import mujoco
import mujoco.viewer
import numpy as np

from so101_mujoco_pid_utils import (
    move_to_pose_pid,
    hold_position_pid,
    build_default_pid,
    DEFAULT_JOINTS,
)

from so101_plotter import JointPlotter


MODEL_PATH = "model/scene_urdf.xml"


# ============================================================
# 🔷 POSES
# ============================================================

starting_position = {
    "shoulder_pan":  -4.4,
    "shoulder_lift": -92.2,
    "elbow_flex":     89.9,
    "wrist_flex":     55.1,
    "wrist_roll":      0.0,
    "gripper":         0.0,
}

desired_zero = {jn: 0.0 for jn in DEFAULT_JOINTS}


# ============================================================
# 🔷 GENERIC PID BUILDER
# ============================================================

def build_pid_controller(kp_base, ki_base, kd_base):

    pid = build_default_pid()

    tau_ref = 10.0

    for jn, gains in pid.gains.items():
        scale = gains.tau_limit / tau_ref

        gains.kp = kp_base * scale
        gains.ki = ki_base * scale
        gains.kd = kd_base * scale

    return pid


# ============================================================
# 🔷 RUN SINGLE EXPERIMENT
# ============================================================

def run_experiment(kp, ki, kd, label):

    print(f"\nRunning {label}")
    print(f"Kp={kp}, Ki={ki}, Kd={kd}")

    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    # Set initial pose
    for jn in DEFAULT_JOINTS:
        jid = m.joint(jn).id
        d.qpos[jid] = np.deg2rad(starting_position[jn])

    mujoco.mj_forward(m, d)

    pid = build_pid_controller(kp, ki, kd)

    plotter = JointPlotter(DEFAULT_JOINTS)

    with mujoco.viewer.launch_passive(m, d) as viewer:

        move_to_pose_pid(
            m, d, viewer,
            desired_zero,
            duration=2.0,
            realtime=True,
            pid=pid,
            plotter=plotter,
            perturb=None  # 🔥 SIN perturbaciones para tuning limpio
        )

        hold_position_pid(
            m, d, viewer,
            desired_zero,
            duration=2.0,
            realtime=True,
            pid=pid,
            plotter=plotter,
            perturb=None
        )

        move_to_pose_pid(
            m, d, viewer,
            starting_position,
            duration=2.0,
            realtime=True,
            pid=pid,
            plotter=plotter,
            perturb=None
        )

        hold_position_pid(
            m, d, viewer,
            starting_position,
            duration=2.0,
            realtime=True,
            pid=pid,
            plotter=plotter,
            perturb=None
        )

    safe_label = label.replace(" ", "_").replace("|", "").replace("=", "")
    plotter.plot(
        title=label,
        save_path=f"results/{safe_label}",
        show=False
    )


# ============================================================
# 🔷 MAIN — FAMILIES DE CONTROLADORES
# ============================================================

if __name__ == "__main__":


    combos_pid = [
        (40, 2, 3),
        (60, 5, 5),
        (60, 8, 8),
        (80, 10, 10),
        (100, 15, 15),
    ]

    for kp, ki, kd in combos_pid:
        run_experiment(
            kp=kp,
            ki=ki,
            kd=kd,
            label=f"PID Controller | Kp={kp} Ki={ki} Kd={kd}"
        )
