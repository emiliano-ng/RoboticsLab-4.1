import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_csv(filename="simulation_log.csv"):
    # 🔥 Preguntar valores PD
    kp = input("Ingrese el valor de Kp: ")
    kd = input("Ingrese el valor de Kd: ")

    # Carpeta automática
    output_folder = f"plots_PD_Kp_{kp}_Kd_{kd}"
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(filename)

    if df.empty:
        print("El archivo está vacío.")
        return

    time = df["time"]

    joints = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll"
    ]

    joint_labels = {
        "shoulder_pan":  "Shoulder Pan",
        "shoulder_lift": "Shoulder Lift",
        "elbow_flex":    "Elbow Flex",
        "wrist_flex":    "Wrist Flex",
        "wrist_roll":    "Wrist Roll",
    }

    for joint in joints:
        if joint not in df.columns:
            print(f"{joint} no está en el CSV")
            continue

        plt.figure()
        plt.plot(time, df[joint])
        plt.xlabel("Time (s)")
        plt.ylabel("Position (deg)")
        plt.title(f"{joint_labels[joint]}  |  PD: Kp={kp}, Kd={kd}")
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(
            output_folder,
            f"{joint}_Kp_{kp}_Kd_{kd}.png"
        )

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Guardado: {save_path}")


if __name__ == "__main__":
    plot_csv("simulation_log.csv")
