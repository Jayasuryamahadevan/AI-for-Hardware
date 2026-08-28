"""A small but honest optical model for the simulated microscope.

The point of simulating physics rather than returning canned arrays is that it
makes the demos *real control problems*. Defocus actually blurs, illumination
actually bleaches, and the camera actually has shot noise — so an agent doing
closed-loop autofocus has to genuinely search, and a badly-planned time-lapse
genuinely destroys the sample. A stub driver would teach an agent nothing and
would validate none of the safety machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Objective name -> (magnification, numerical aperture, working distance µm)
OBJECTIVES: dict[str, tuple[float, float, float]] = {
    "4x":  (4.0,  0.13, 17000.0),
    "10x": (10.0, 0.30, 5200.0),
    "20x": (20.0, 0.45, 2100.0),
    "40x": (40.0, 0.65, 600.0),
    "63x": (63.0, 1.40, 190.0),
}

#: Channel -> (excitation nm, emission nm, bleach rate per J, display name)
CHANNELS: dict[str, tuple[float, float, float, str]] = {
    "BF":     (550.0, 550.0, 0.0,   "Brightfield"),
    "DAPI":   (358.0, 461.0, 0.055, "DAPI / nuclei"),
    "FITC":   (488.0, 519.0, 0.030, "FITC / GFP"),
    "TRITC":  (557.0, 576.0, 0.018, "TRITC / RFP"),
    "Cy5":    (650.0, 670.0, 0.009, "Cy5 / far red"),
}

CAMERA_PIXEL_UM = 6.5  # physical sensor pitch, typical sCMOS


@dataclass
class Emitter:
    """One fluorescent object -- a cell nucleus, a bead, a punctum.

    Kept as a value type for readability at the call site. The specimen itself
    stores emitters as parallel numpy arrays, not as a million of these: a
    20 mm slide at realistic density is nearly a million objects, and building
    that many Python objects took tens of seconds at gateway startup.
    """

    x_um: float
    y_um: float
    z_um: float
    radius_um: float
    #: Per-channel brightness in arbitrary photon units.
    brightness: dict[str, float] = field(default_factory=dict)
    #: Fraction of fluorophore remaining; decays with cumulative exposure.
    bleach: float = 1.0


class Specimen:
    """A synthetic slide: emitters scattered over a slightly tilted focal plane.

    The tilt is deliberate. A perfectly flat sample makes autofocus trivial and
    hides the failure mode that matters in real tiling -- focus drift across the
    field. With tilt, an agent that autofocuses once and then tiles five
    millimetres produces visibly bad images, which is exactly the lesson worth
    simulating.

    Emitters live in parallel numpy arrays rather than as objects. At a
    realistic 3000 cells/mm^2 over a 20 mm slide that is nearly a million
    emitters, and constructing them one at a time made the gateway take half a
    minute to start. Vectorised, the same slide is built in milliseconds and
    a field lookup is a single boolean mask.
    """

    def __init__(
        self,
        *,
        seed: int = 7,
        extent_um: float = 2000.0,
        density_per_mm2: float = 3000.0,
        focal_plane_z: float = 100.0,
        tilt_um_per_mm: float = 1.8,
        colony_sigma_um: float = 70.0,
        cells_per_colony: int = 25,
        centre_um: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        # Density is specified per mm^2 rather than as a raw count because that
        # is the number that has to be right: a 20x field is only ~83 um across,
        # so a count chosen for the whole slide leaves every field empty.
        self.rng = np.random.default_rng(seed)
        self.extent_um = extent_um
        self.focal_plane_z = focal_plane_z
        self.tilt = tilt_um_per_mm
        # Where the slide sits in *stage* coordinates. A specimen centred on
        # zero while the stage travels 0..20000 um puts the instrument's home
        # position at the extreme corner of the slide, which looks like a
        # broken autofocus rather than an empty field.
        self.centre_um = centre_um
        cx, cy = centre_um
        half = extent_um / 2

        area_mm2 = (extent_um / 1000.0) ** 2
        count = max(1, int(density_per_mm2 * area_mm2))
        # Cells clump; uniform scatter looks wrong and behaves wrong under
        # segmentation, so seed colonies and jitter around them.
        colonies = max(1, count // cells_per_colony)
        centres = self.rng.uniform(-half, half, size=(colonies, 2)) + np.array([cx, cy])
        assignment = np.arange(count) % colonies

        self.x = np.clip(
            centres[assignment, 0] + self.rng.normal(0, colony_sigma_um, count),
            cx - half, cx + half,
        )
        self.y = np.clip(
            centres[assignment, 1] + self.rng.normal(0, colony_sigma_um, count),
            cy - half, cy + half,
        )
        self.z = self.surface_z_array(self.x, self.y) + self.rng.normal(0, 1.2, count)
        self.radius = np.abs(self.rng.normal(1.4, 0.5, count)) + 0.35
        #: Fraction of fluorophore remaining per emitter.
        self.bleach = np.ones(count)
        #: Integrated photon budget per emitter, per channel -- not a peak
        #: value: `render` conserves total intensity as blur widens, so these
        #: must be large enough to stay above shot noise when spread over a
        #: defocused disc.
        self.brightness: dict[str, np.ndarray] = {
            "BF": np.full(count, 60_000.0),
            "DAPI": np.abs(self.rng.normal(75_000, 18_000, count)),
            "FITC": np.abs(self.rng.normal(52_000, 22_000, count)),
            "TRITC": np.abs(self.rng.normal(31_000, 14_000, count)),
            "Cy5": np.abs(self.rng.normal(18_000, 9_000, count)),
        }

    @property
    def count(self) -> int:
        return int(self.x.size)

    def surface_z_array(self, x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
        dx = (x_um - self.centre_um[0]) / 1000.0
        dy = (y_um - self.centre_um[1]) / 1000.0
        return self.focal_plane_z + self.tilt * dx * 0.8 + self.tilt * dy * 0.5

    def surface_z(self, x_um: float, y_um: float) -> float:
        """True in-focus Z at a given stage position, including tilt.

        Tilt is measured from the slide centre, so the nominal focal plane is
        the height at the centre rather than at stage zero.
        """
        dx = (x_um - self.centre_um[0]) / 1000.0
        dy = (y_um - self.centre_um[1]) / 1000.0
        return self.focal_plane_z + self.tilt * dx * 0.8 + self.tilt * dy * 0.5

    def near_indices(self, x_um: float, y_um: float, radius_um: float) -> np.ndarray:
        """Indices of emitters within `radius_um` of a stage position."""
        if not self.count:
            return np.empty(0, dtype=np.intp)
        # A cheap bounding-box reject before the hypot: at a million emitters
        # the square root is the expensive part and almost every emitter fails.
        box = (
            (np.abs(self.x - x_um) <= radius_um) & (np.abs(self.y - y_um) <= radius_um)
        )
        candidates = np.nonzero(box)[0]
        if not candidates.size:
            return candidates
        dx = self.x[candidates] - x_um
        dy = self.y[candidates] - y_um
        return candidates[(dx * dx + dy * dy) <= radius_um * radius_um]

    def near(self, x_um: float, y_um: float, radius_um: float) -> list[Emitter]:
        """Emitters near a position, as value objects.

        Convenient for inspection and tests. `render` uses `near_indices` and
        the arrays directly, because materialising objects per frame would
        undo the point of storing them as arrays.
        """
        return [
            Emitter(
                x_um=float(self.x[i]), y_um=float(self.y[i]), z_um=float(self.z[i]),
                radius_um=float(self.radius[i]),
                brightness={c: float(v[i]) for c, v in self.brightness.items()},
                bleach=float(self.bleach[i]),
            )
            for i in self.near_indices(x_um, y_um, radius_um)
        ]


def depth_of_field_um(wavelength_nm: float, na: float, n: float = 1.0) -> float:
    """Classical DOF. Sets how sharply image quality falls off with defocus."""
    return (wavelength_nm * 1e-3 * n) / (na * na) if na > 0 else 1e6


def render(
    specimen: Specimen,
    *,
    x_um: float,
    y_um: float,
    z_um: float,
    objective: str,
    channel: str,
    exposure_ms: float,
    intensity_pct: float,
    width: int,
    height: int,
    bit_depth: int = 16,
    rng: np.random.Generator | None = None,
    apply_bleaching: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render one frame and return it with per-frame quality metrics.

    Returns (image, metrics) where metrics carry the focus score an autofocus
    routine would compute, plus the ground-truth defocus for scoring the agent.
    """
    rng = rng or np.random.default_rng()
    mag, na, _wd = OBJECTIVES[objective]
    _ex_nm, em_nm, bleach_rate, _label = CHANNELS[channel]

    # Sampling: sensor pitch projected back into specimen space.
    px_um = CAMERA_PIXEL_UM / mag
    fov_w = width * px_um
    fov_h = height * px_um

    dof = depth_of_field_um(em_nm, na)
    ideal_z = specimen.surface_z(x_um, y_um)
    defocus = z_um - ideal_z
    # Defocus blur, in specimen µm, then converted to pixels.
    sigma_um = 0.55 * abs(defocus) * na + 0.20
    sigma_px = max(0.6, sigma_um / px_um)

    img = np.zeros((height, width), dtype=np.float64)
    photon_scale = (exposure_ms / 100.0) * (intensity_pct / 100.0)

    # Grid in specimen coordinates for this field.
    ys = (np.arange(height) - height / 2) * px_um + y_um
    xs = (np.arange(width) - width / 2) * px_um + x_um

    visible = specimen.near_indices(x_um, y_um, radius_um=max(fov_w, fov_h))
    channel_brightness = specimen.brightness.get(channel)
    for i in visible:
        base = float(channel_brightness[i]) if channel_brightness is not None else 0.0
        if apply_bleaching:
            base *= float(specimen.bleach[i])
        if base <= 0:
            continue
        # Axial response: emitters off the focal plane contribute less and wider.
        dz = z_um - float(specimen.z[i])
        axial = 1.0 / (1.0 + (dz / max(dof, 0.3)) ** 2)
        eff_sigma_px = math.hypot(sigma_px, float(specimen.radius[i]) / px_um)
        if eff_sigma_px > 60:  # too diffuse to matter; skip for speed
            continue
        cx = (float(specimen.x[i]) - xs[0]) / px_um
        cy = (float(specimen.y[i]) - ys[0]) / px_um
        r = int(min(4 * eff_sigma_px, 80))
        x0, x1 = max(0, int(cx) - r), min(width, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(height, int(cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        gx = np.arange(x0, x1) - cx
        gy = np.arange(y0, y1) - cy
        g = np.exp(-(gy[:, None] ** 2 + gx[None, :] ** 2) / (2 * eff_sigma_px ** 2))
        amp = base * axial * photon_scale / (2 * math.pi * eff_sigma_px ** 2)
        img[y0:y1, x0:x1] += amp * g

    if channel == "BF":
        # Brightfield: bright background, emitters absorb.
        img = 1200.0 * photon_scale - img * 0.9
        img = np.clip(img, 0, None)
    else:
        img += 12.0 * photon_scale  # autofluorescence / stray light

    # Shot noise then read noise — the order matters physically.
    img = rng.poisson(np.clip(img, 0, 1e7)).astype(np.float64)
    img += rng.normal(0.0, 2.4, size=img.shape)
    img += 100.0  # camera offset

    max_val = (1 << bit_depth) - 1
    saturated = float(np.mean(img >= max_val))
    out = np.clip(img, 0, max_val).astype(np.uint16 if bit_depth > 8 else np.uint8)

    # Bleach what we just illuminated.
    if apply_bleaching and bleach_rate > 0:
        dose = photon_scale * bleach_rate
        illuminated = specimen.near_indices(x_um, y_um, radius_um=max(fov_w, fov_h) / 2)
        if illuminated.size:
            specimen.bleach[illuminated] = np.maximum(
                0.02, specimen.bleach[illuminated] * math.exp(-dose)
            )

    metrics = {
        "focus_score": focus_score(out),
        "defocus_um": float(defocus),
        "depth_of_field_um": float(dof),
        "pixel_size_um": float(px_um),
        "fov_um": float(fov_w),
        "mean": float(out.mean()),
        "p99": float(np.percentile(out, 99)),
        "saturated_fraction": saturated,
        # Signal against the *noise floor*, not against the overall spread.
        # Dividing by out.std() would be scale-invariant: bleaching halves both
        # the signal and the spread, so the ratio would not move and an agent
        # watching this number would report healthy signal while the specimen
        # burned away. Estimating the noise separately is what makes the decay
        # visible.
        "snr_estimate": float(
            (np.percentile(out, 99) - np.median(out)) / max(1.0, estimate_noise_sigma(out))
        ),
    }
    return out, metrics


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Immerkaer's noise estimate: robust to image content, cheap to compute.

    Convolving with [[1,-2,1],[-2,4,-2],[1,-2,1]] annihilates any locally
    quadratic surface, so smooth structure and blurred blobs contribute almost
    nothing and what survives is noise. That property is the whole point here:
    the focus metric needs to know the noise floor *without* knowing whether it
    is looking at a sharp field or an empty one.
    """
    a = img.astype(np.float64)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    response = (
        4 * a[1:-1, 1:-1]
        - 2 * (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
        + a[:-2, :-2] + a[:-2, 2:] + a[2:, :-2] + a[2:, 2:]
    )
    # sqrt(pi/2) / 6 converts mean absolute response into a sigma estimate.
    return float(np.sqrt(math.pi / 2) * np.abs(response).mean() / 6.0)


def focus_score(img: np.ndarray) -> float:
    """Noise-corrected normalised variance -- the autofocus metric.

    Two metrics were tried before this one, and both failed in ways worth
    recording, because both are the textbook answer:

    *Variance of the Laplacian* pins at a constant on an empty field: for white
    noise the five-point Laplacian has variance 20 sigma^2 against a signal
    energy of sigma^2, so the ratio sits at 20 no matter where the focus drive
    is. Subtracting the noise contribution fixes the empty field but leaves a
    ratio of two near-zero quantities off focus, which is numerically wild --
    it produced scores of 44 at fifteen micrometres of defocus and 0.1 at true
    focus, so a hill-climb walked away from the specimen.

    Normalised variance is stable because the denominator stays bounded. The
    optical model conserves total flux as defocus blur widens, so the mean of
    the background-subtracted field is roughly constant while its variance
    falls as the signal spreads -- which makes var/mean^2 peak sharply at
    focus. Dividing by the square of the mean is also what keeps lamp power
    from dominating: doubling the illumination doubles the mean and quadruples
    the variance, so the ratio is invariant to the multiplicative part. It is
    not perfectly flat -- measured at 1.5x for a doubling - because more
    photons genuinely do recover more detail against Poisson noise, and that is
    a real improvement rather than an artefact. What matters for a search is
    that the *location* of the peak does not move with lamp power, and it does
    not.

    The noise floor is estimated per frame and subtracted, and the denominator
    is floored at the noise level, so an empty field scores zero instead of
    dividing two pieces of noise by each other.
    """
    a = img.astype(np.float64)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    sigma = estimate_noise_sigma(a)
    # Remove the camera offset and background pedestal. Without this the metric
    # is dominated by a large constant that carries no focus information.
    a = a - np.median(a)
    signal_variance = float(a.var()) - sigma * sigma
    if signal_variance <= 0.0:
        return 0.0  # nothing above the noise floor: nothing to focus on
    # Flooring at sigma keeps a nearly-empty field from dividing by ~0.
    scale = max(float(np.abs(a).mean()), sigma)
    return signal_variance / (scale * scale)
