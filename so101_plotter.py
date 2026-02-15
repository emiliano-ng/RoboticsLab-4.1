import os
import numpy as np
import matplotlib.pyplot as plt


class JointPlotter:

    def __init__(self, joint_names):

        self.joint_names = joint_names

        self.time_history = []

        self.q_history = {jn: [] for jn in joint_names}
        self.q_des_history = {jn: [] for jn in joint_names}

    # ============================================================
    # 🔷 SAMPLE
    # ============================================================

    def sample(self, m, d, q_des_current=None):

        t = float(d.time)
        self.time_history.append(t)

        for jn in self.joint_names:

            jid = m.joint(jn).id
            q_val = d.qpos[jid]

            self.q_history[jn].append(q_val)

            if q_des_current is not None:
                self.q_des_history[jn].append(q_des_current[jn])
            else:
                self.q_des_history[jn].append(np.nan)

    # ============================================================
    # 🔷 PLOT + SAVE
    # ============================================================

    def plot(self, title="", save_path=None, show=False):

        # Crear carpeta si no existe
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for joint in self.joint_names:

            plt.figure()

            t = self.time_history
            q = self.q_history[joint]
            q_des = self.q_des_history[joint]

            plt.plot(t, q, label="q")
            plt.plot(t, q_des, "--", label="q_des")

            plt.xlabel("Time [s]")
            plt.ylabel("Position [rad]")
            plt.title(f"{title} | {joint}")
            plt.legend()
            plt.grid(True)

            if save_path is not None:
                filename = f"{save_path}_{joint}.png"
                plt.savefig(filename, dpi=300)

            if show:
                plt.show()

            plt.close()  # 🔥 IMPORTANTE para no bloquear
