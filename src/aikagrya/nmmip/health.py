from dataclasses import dataclass


@dataclass
class Health:
    delta: float
    r_fix: float
    eigen_residual: float
    entropy: float
    rho: float                # canonical: Var(x) * d
    participation: float      # PR / d
    uniformity_cosine: float
    converged: bool
    steps: int
    t_rec: int

    def passes_lenient(self) -> bool:
        return (
            self.delta < 1e-5 and
            self.r_fix <= 5e-7 and
            self.rho >= 0.30 and
            self.participation >= 0.30 and
            self.uniformity_cosine <= 0.08 and
            self.converged and
            self.t_rec <= 250
        )

    def passes_strict(self) -> bool:
        return (
            self.delta < 1e-6 and
            self.r_fix <= 1e-9 and
            self.rho >= 0.70 and
            self.participation >= 0.30 and
            self.uniformity_cosine <= 0.08 and
            self.converged and
            self.t_rec <= 200
        )


