import csv, os, cv2, argparse
import numpy as np
from so2_objective import *
from RDueling import *
from utils import *
import matplotlib.pyplot as plt



def load_horizon_points(csv_path, prefix="0006/"):
    horizon_dict = {}
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith(prefix+ "/"):
                fname = os.path.basename(row[0])
                x1, y1, x2, y2 = map(float, row[-4:])
                horizon_dict[fname] = (x1, y1, x2, y2)
    return horizon_dict


def rotate_image(img, theta_deg):
    """Rotate image by theta degrees (positive = CCW)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), theta_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    # rotated = cv2.warpAffine(img, M, (w, h),
    #                         flags=cv2.INTER_LINEAR,
    #                         borderMode=cv2.BORDER_CONSTANT,
    #                         borderValue=(0, 0, 0))  # black
    return rotated


def main():
    parser = argparse.ArgumentParser(
        description="Run Dueling Optimization on SO(2) for horizon correction."
    )
    parser.add_argument("--min_deg", type=float, default=15.0,
                        help="Minimum absolute degree for image selection (default: 15°)")
    parser.add_argument("--max_deg", type=float, default=90.0,
                        help="Maximum absolute degree for image selection (default: 30°)")
    args = parser.parse_args()

    min_deg = args.min_deg
    max_deg = args.max_deg


    # Dataset parameters
    DATA_ROOT = "hlw"
    DATAPATH = os.path.join(DATA_ROOT, "metadata.csv")
    FOLDER_PREFIX = "0006"

    # Load data and build dictionary
    horizon_dict = load_horizon_points(DATAPATH, FOLDER_PREFIX)
    folder = os.path.join(DATA_ROOT, "images", FOLDER_PREFIX.strip("/"))
    files = os.listdir(folder)

    # Sort in ascending order of absolute angle
    angles_list = []
    for fname in files:
        if fname not in horizon_dict:
            continue
        coords = horizon_dict[fname]
        theta = horizon_angle(coords)
        abs_deg = abs(math.degrees(theta))
        angles_list.append((fname, abs_deg, theta))
    angles_list.sort(key=lambda x: x[1]) 

    # Select subsets of images with noticeable degree
    selected_list = [(fname, deg, theta) for fname, deg, theta in angles_list if min_deg <= deg <= max_deg]
    print(f"Number of images in [{min_deg}°, {max_deg}°]: {len(selected_list)}")
    for item in selected_list:
        print(item)
    
    filename = selected_list[0][0]

    # # Show image
    # img_path = os.path.join(DATA_ROOT, "images", FOLDER_PREFIX, filename)
    # img = cv2.imread(img_path)
    # if img is None:
    #     raise FileNotFoundError(f"Cannot open image at {img_path}")
    # cv2.imshow("Selected Image", img)
    # print("Press any key in the image window to continue...")
    # cv2.waitKey(0)           
    # cv2.destroyAllWindows()  

    oracle = SO2Objective.from_dict(horizon_dict, filename)
    print("\nConstructed oracle object:")
    print("R_star =\n", oracle.R_star)

    R0 = angle_to_so2(np.deg2rad(0.0))
    optimizer = RDueling(oracle, lr=1e-2, delta=1e-6, T=100)
    log = optimizer.run(R0)

    final_theta_deg = np.degrees(so2_to_angle(log["R_final"]))
    best_theta_deg = np.degrees(so2_to_angle(log["R_best"]))

    print("\n=======================================")
    print(f"[Result] Final angle (deg): {final_theta_deg:.2f}")
    print(f"[Result] Best-so-far angle (deg): {best_theta_deg:.2f}")
    print(f"[Result] f_best: {log['f_best']:.6f}")
    print("=======================================")

    plt.figure(figsize=(6,4))
    plt.plot(log["f_seq"], label="f(R_t)")
    plt.plot(log["f_best_seq"], label="best so far", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("f value")
    plt.legend()
    plt.title("Dueling Optimization on SO(2)")

    # Compute the optimal correction angle (in degrees)
    theta_best_deg = so2_to_angle(log["R_best"], degrees=True)

    # Reconstruct full image path
    img_path = os.path.join(DATA_ROOT, "images", FOLDER_PREFIX, filename)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image at {img_path}")
    theta_best_deg = so2_to_angle(log["R_best"], degrees=True) # Compute rotation and visualize
    img_rotated = rotate_image(img, theta_best_deg)

    # Visualize before and after
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    plt.title(f"Rotated Image ({theta_best_deg:.2f}°)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()