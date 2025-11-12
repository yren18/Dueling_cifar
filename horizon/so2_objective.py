from utils import *

class SO2Objective:
    def __init__(self, theta_true):
        """
        Initialize with a fixed true horizon angle.
        Ground-truth optimal rotation is R* = R(-theta_true).
        """
        self.optimal_degree = -theta_true
        self.R_star = angle_to_so2(-theta_true)

    @classmethod
    def from_dict(cls, horizon_dict, image_name):
        """
        Alternate constructor:
        Build an SO2Objective instance from a preloaded horizon_dict.
        """
        import math
        if image_name not in horizon_dict:
            raise ValueError(f"{image_name} not found in horizon_dict.")

        x1, y1, x2, y2 = horizon_dict[image_name]
        dx, dy = x2 - x1, y2 - y1
        theta_true = math.atan2(dy, dx)
        return cls(theta_true)

    def f(self, R):
        """Compute f(R) = 0.5 * ||Log(Rᵀ R*)||²"""
        R_rel = R.T @ self.R_star
        omega = so2_to_angle(R_rel, degrees=False)
        omega = float(np.squeeze(omega))
        return 0.5 * omega**2
    

    def exp(self, R, v):
        """
        Exponential map on SO(2):
        v is a 2x2 tangent matrix of the form RΩω.
        """
        Omega = np.array([[0, -1], [1, 0]])
        omega = np.trace(Omega.T @ (R.T @ v)) / 2.0

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