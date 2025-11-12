import numpy as np
from utils import so2_to_angle  
import so2_objective             


class RDueling:
    def __init__(self, oracle, lr=1e-2, delta=0.05, T=100):
        """
        oracle: an object with .f(R), .exp(R, omega), .sample_unit_tangent(R)
        lr: stepsize
        delta: pertubation scale
        T: number of iterations
        """
        self.oracle = oracle
        self.lr = lr
        self.delta = delta
        self.T = T

    def run(self, R0):
        """
        Run Riemannian dueling optimization with initial point R0.
        """
        R_current = R0
        theta_seq = []      # store angle trajectory
        f_seq = []          # store raw function values
        f_best_seq = []     # store best-so-far values

        # Initialization
        f_best = self.oracle.f(R_current)
        R_best = R_current
        theta_seq.append(so2_to_angle(R_current))
        f_seq.append(f_best)
        f_best_seq.append(f_best)

        for t in range(self.T):
            # Sample random unit tangent vector
            v = self.oracle.sample_unit_tangent(R_current)

            # Evaluate dueling oracle
            R_plus = self.oracle.exp(R_current, self.delta * v)
            R_minus = self.oracle.exp(R_current, -self.delta * v)
            f_plus = self.oracle.f(R_plus)
            f_minus = self.oracle.f(R_minus)
            o = np.sign(f_plus - f_minus)

            # Update rotation
            R_current = self.oracle.exp(R_current, -self.lr * o * v)

            # Evaluate and record
            f_val = self.oracle.f(R_current)
            theta_seq.append(so2_to_angle(R_current))
            f_seq.append(f_val)

            # Track best-so-far (dueling version)
            if self.oracle.f(R_best) > self.oracle.f(R_current):
                R_best = R_current
                f_best = f_val

            f_best_seq.append(f_best)

        # Return output
        return {
            "R_final": R_current,
            "R_best": R_best,
            "f_best": f_best,
            "theta_seq": np.array(theta_seq),
            "f_seq": np.array(f_seq),
            "f_best_seq": np.array(f_best_seq),
        }
