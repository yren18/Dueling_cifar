import csv, os
import math
import numpy as np


def load_horizon_points(csv_path, prefix="0006/"):
    horizon_dict = {}
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith(prefix):
                fname = os.path.basename(row[0])
                x1, y1, x2, y2 = map(float, row[-4:])
                horizon_dict[fname] = (x1, y1, x2, y2)
    return horizon_dict


class SO2Objective:
    def __init__(self, theta_true):
        """
        Initialize with a fixed true horizon angle.
        Ground-truth optimal rotation is R* = R(-theta_true).
        """
        self.R_star = angle_to_so2(-theta_true)

    def f(self, R):
        """Compute f(R) = 0.5 * ||Log(Rᵀ R*)||²"""
        R_rel = R.T @ self.R_star
        omega = so2_log(R_rel)
        return 0.5 * omega**2

    def exp(self, R, omega):
        """
        Exponential map on SO(2):
        Move from R along tangent vector ω.
        """
        return R @ so2_exp(omega)

    def sample_unit_tangent(self, R):
        """
        Uniformly sample a unit tangent vector at R ∈ SO(2).
        Since SO(2) is 1-D, tangent direction is ±RΩ.
        """
        Omega = np.array([[0, -1],
                          [1,  0]])
        sign = np.random.choice([-1.0, 1.0])
        return sign * (R @ Omega)




class DuelingOptimizer:
    def __init__(self, oracle, lr=0.1, delta=0.05, T=100):
        """
        oracle: an object with .f(R), .exp(R, omega), .sample_unit_tangent(R)
        lr: learning rate for updates
        delta: exploration step size
        T: number of iterations
        """
        self.oracle = oracle
        self.lr = lr
        self.delta = delta
        self.T = T

    def run(self, R0):
        """Run dueling optimization starting from R0."""
        R = R0
        history = [R0]

        for t in range(self.T):
            # 1. Sample a random unit tangent vector
            V = self.oracle.sample_unit_tangent(R)

            # 2. Move slightly along +delta and -delta directions
            R_plus = self.oracle.exp(R, self.delta)
            R_minus = self.oracle.exp(R, -self.delta)

            # 3. Compare via oracle feedback (dueling signal)
            f_plus = self.oracle.f(R_plus)
            f_minus = self.oracle.f(R_minus)

            # Dueling feedback: prefer smaller f
            g = -np.sign(f_plus - f_minus)  # +1 means move toward better side

            # 4. Update: move along tangent direction scaled by feedback
            R = self.oracle.exp(R, -self.lr * g * self.delta)
            history.append(R)

        return history



# Dataset parameters
DATA_ROOT = "hlw"
DATAPATH = os.path.join(DATA_ROOT, "metadata.csv")
FOLDER_PREFIX = "0006/"


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
    angles_list.append((fname, abs_deg))
angles_list.sort(key=lambda x: x[1]) 

# Select subsets of images with noticeable degree
min_deg, max_deg = 20, 40
selected_list = [(fname, deg) for fname, deg in angles_list if min_deg <= deg <= max_deg]
print(f"Number of images in [{min_deg}°, {max_deg}°]: {len(selected_list)}")
for item in selected_list:
    print(item)





# open hlw/images/0006/3436779341_8627c0524a_o.jpg
# open hlw/images/0006/

