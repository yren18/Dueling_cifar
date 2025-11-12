import csv, os, cv2, argparse, math
import numpy as np
import matplotlib.pyplot as plt
from so2_objective import *
from RDueling import *
from utils import *


def load_horizon_points(csv_path, prefix="0006/"):
    """Load horizon line coordinates from CSV."""
    horizon_dict = {}
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith(prefix + "/"):
                fname = os.path.basename(row[0])
                x1, y1, x2, y2 = map(float, row[-4:])
                horizon_dict[fname] = (x1, y1, x2, y2)
    return horizon_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run Dueling Optimization on SO(2) for horizon correction."
    )
    parser.add_argument("--min_deg", type=float, default=15.0,
                        help="Minimum absolute degree for image selection (default: 15°)")
    parser.add_argument("--max_deg", type=float, default=90.0,
                        help="Maximum absolute degree for image selection (default: 90°)")
    parser.add_argument("--save", action="store_true",
                        help="If set, save result figures to ./result/ (default: False)")
    parser.add_argument("--show", type=lambda x: x.lower() == 'true',
                        default=True,
                        help="Whether to display plots (True/False, default: True)")
    args = parser.parse_args()
    min_deg = args.min_deg
    max_deg = args.max_deg


    # Dataset parameters
    DATA_ROOT = "hlw"
    DATAPATH = os.path.join(DATA_ROOT, "metadata.csv")
    FOLDER_PREFIX = "0006"

    # Load data
    horizon_dict = load_horizon_points(DATAPATH, FOLDER_PREFIX)
    folder = os.path.join(DATA_ROOT, "images", FOLDER_PREFIX.strip("/"))
    files = os.listdir(folder)

    # Sort by angle magnitude
    angles_list = []
    for fname in files:
        if fname not in horizon_dict:
            continue
        coords = horizon_dict[fname]
        theta = horizon_angle(coords)
        abs_deg = abs(math.degrees(theta))
        angles_list.append((fname, abs_deg, theta))
    angles_list.sort(key=lambda x: x[1])

    # Select subset
    selected_list = [(fname, deg, theta) for fname, deg, theta in angles_list if min_deg <= deg <= max_deg]
    # print(f"Number of images in [{min_deg}°, {max_deg}°]: {len(selected_list)}")
    # for item in selected_list:
    #     print(item)

    filename = selected_list[0][0]  # pick first image
    base_name = os.path.splitext(filename)[0]  # e.g. 2680766258_66b5a9fcbf_o

    # Build oracle and optimizer
    oracle = SO2Objective.from_dict(horizon_dict, filename)
    print("\nConstructed oracle object:")
    print("R_star =\n", oracle.R_star)

    R0 = angle_to_so2(np.deg2rad(0.0))
    optimizer = RDueling(oracle, lr=1e-2, delta=1e-6, T=100)
    log = optimizer.run(R0)

    final_theta_deg = np.degrees(so2_to_angle(log["R_final"]))
    best_theta_deg = np.degrees(so2_to_angle(log["R_best"]))

    print("\n=======================================")
    print(f"[Result] True horizon angle (deg): {np.degrees(oracle.optimal_degree):.2f}")
    print(f"[Result] Final angle (deg): {final_theta_deg:.2f}")
    print(f"[Result] Best-so-far angle (deg): {best_theta_deg:.2f}")
    print(f"[Result] f_best: {log['f_best']:.6f}")
    print("=======================================")

    # Plot and save result
    RESULT_DIR = os.path.join(os.getcwd(), "result")
    if args.save:
        os.makedirs(RESULT_DIR, exist_ok=True)

    # Plot Loss curve
    fig1 = plt.figure(figsize=(6, 4))
    # plt.plot(log["f_seq"], label="f(R_t)")
    plt.plot(log["f_best_seq"], label="RDueling", linestyle="-",linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel(r"$f(R_t) - f^*$")
    plt.yscale("log") 
    plt.legend()
    plt.title(f"Figure 1: Loss vs Iteration ({base_name})")

    if args.save:
        fig1_path = os.path.join(RESULT_DIR, f"{base_name}_loss_vs_iteration.png")
        fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] Figure 1 → {fig1_path}")

    # Plot Original vs Rotated
    img_path = os.path.join(DATA_ROOT, "images", FOLDER_PREFIX, filename)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image at {img_path}")
    theta_best_deg = so2_to_angle(log["R_best"], degrees=True)
    img_rotated = rotate_image(img, theta_best_deg)

    fig2 = plt.figure(figsize=(10, 4))
    plt.suptitle(f"Figure 2: Horizon Correction Result ({base_name})", fontsize=12)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    plt.title(f"Rotated Image ({theta_best_deg:.2f}°)")
    plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        fig2_path = os.path.join(RESULT_DIR, f"{base_name}_before_and_after.png")
        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] Figure 2 → {fig2_path}")
    if args.show:
        plt.show()
    else:
        plt.close('all')


if __name__ == "__main__":
    main()
