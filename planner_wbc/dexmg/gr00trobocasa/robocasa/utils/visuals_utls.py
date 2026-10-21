from dataclasses import dataclass

import numpy as np

from robocasa.models.objects.objects import MJCFObject


@dataclass(frozen=True)
class Gradient:
    rgba_a: np.ndarray = np.array([0, 0, 0, 1])
    rgba_b: np.ndarray = np.array([1, 1, 1, 1])

    def sample_linear(self, t: float) -> np.ndarray:
        return (1 - t) * self.rgba_a + t * self.rgba_b

    def sample_per_channel(self, t: tuple[float, float, float, float]):
        return [np.interp(t[i], [0, 1], [self.rgba_a[i], self.rgba_b[i]]) for i in range(4)]


def randomize_materials_rgba(
    rng: np.random.Generator, mjcf_obj: MJCFObject, gradient: Gradient, linear: bool
):
    for asset in mjcf_obj.asset:
        if asset.tag == "material":
            if linear:
                t = rng.uniform(0, 1)
                rgba = gradient.sample_linear(t)
            else:
                t = rng.uniform(low=[0, 0, 0, 1], high=[1, 1, 1, 1]).tolist()
                rgba = gradient.sample_per_channel(t)
            asset.set("rgba", f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {rgba[3]:.3f}")
