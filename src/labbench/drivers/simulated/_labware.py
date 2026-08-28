"""Shared labware: plates that exist independently of any one instrument.

The point of modelling this separately is integration. A plate the liquid
handler dispenses into must be the same plate the reader measures and the same
plate the incubator warms -- otherwise each simulated instrument is its own
sealed fiction, and an agent can never be caught by the error that matters
most in a real workflow: reading a plate it never actually filled.

So plates live in a process-wide store keyed by barcode. That store is the
simulation's stand-in for a bench: putting a plate somewhere and carrying it
between instruments. Real labs solve this with a plate hotel and a robot arm;
here a barcode is enough.

The chemistry is deliberately shallow but not fake. Wells conserve volume,
mixing dilutes correctly by mass balance, evaporation is a function of
temperature and time, and a fluorophore's signal is proportional to the amount
actually present. That is enough to make a serial dilution either right or
visibly wrong.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

#: Plate geometries, as (rows, columns, working volume uL, dead volume uL).
FORMATS: dict[str, tuple[int, int, float, float]] = {
    "6":    (2, 3, 3000.0, 200.0),
    "24":   (4, 6, 1000.0, 50.0),
    "96":   (8, 12, 200.0, 10.0),
    "384":  (16, 24, 50.0, 5.0),
    "1536": (32, 48, 8.0, 1.0),
}

#: Reagents the simulated lab knows about, with the optics needed to read them.
#: (molar extinction at its peak, excitation nm, emission nm, quantum yield)
REAGENTS: dict[str, tuple[float, float | None, float | None, float]] = {
    "water":       (0.0, None, None, 0.0),
    "buffer":      (0.0, None, None, 0.0),
    "dye_red":     (18000.0, None, None, 0.0),      # absorbance only
    "fluorescein": (76900.0, 494.0, 512.0, 0.95),
    "rhodamine":   (95000.0, 555.0, 580.0, 0.70),
    "resazurin":   (12000.0, 560.0, 590.0, 0.11),   # viability indicator
    "protein":     (43800.0, 280.0, 340.0, 0.05),
    "cells":       (0.0, None, None, 0.0),
}


def well_name(row: int, column: int) -> str:
    """(0, 0) -> 'A1'. Rows past Z wrap to AA, as 1536-well plates require."""
    if row < 26:
        letter = chr(ord("A") + row)
    else:
        letter = chr(ord("A") + row // 26 - 1) + chr(ord("A") + row % 26)
    return f"{letter}{column + 1}"


def parse_well(name: str) -> tuple[int, int]:
    """'A1' -> (0, 0). Raises ValueError on anything unparseable."""
    name = name.strip().upper()
    letters = "".join(c for c in name if c.isalpha())
    digits = "".join(c for c in name if c.isdigit())
    if not letters or not digits:
        raise ValueError(f"{name!r} is not a well name like 'A1' or 'AB12'")
    row = 0
    for char in letters:
        row = row * 26 + (ord(char) - ord("A") + 1)
    return row - 1, int(digits) - 1


@dataclass
class Well:
    """One well. Composition is tracked as absolute amounts, not concentrations.

    Amounts rather than concentrations because that is what survives a
    transfer: moving 10 uL of a well moves a tenth of everything in it, and
    expressing that in concentrations invites the classic dilution error.
    """

    volume_ul: float = 0.0
    #: Reagent name -> picomoles present.
    contents: dict[str, float] = field(default_factory=dict)

    def concentration_um(self, reagent: str) -> float:
        """Micromolar. pmol / uL is exactly uM, which is why amounts are in pmol."""
        if self.volume_ul <= 0:
            return 0.0
        return self.contents.get(reagent, 0.0) / self.volume_ul

    def add(self, volume_ul: float, contents: dict[str, float]) -> None:
        self.volume_ul += volume_ul
        for reagent, pmol in contents.items():
            self.contents[reagent] = self.contents.get(reagent, 0.0) + pmol

    def remove(self, volume_ul: float) -> dict[str, float]:
        """Take a volume out and return what came with it.

        Removal is proportional: aspirating a fifth of a well takes a fifth of
        every solute. Assuming a well is homogeneous is a simplification, and a
        deliberate one -- modelling incomplete mixing would be more realistic
        and would teach an agent nothing it can act on.
        """
        if volume_ul <= 0:
            return {}
        taken = min(volume_ul, self.volume_ul)
        fraction = taken / self.volume_ul if self.volume_ul > 0 else 0.0
        moved = {r: pmol * fraction for r, pmol in self.contents.items()}
        for reagent in list(self.contents):
            self.contents[reagent] -= moved[reagent]
            if self.contents[reagent] <= 1e-12:
                del self.contents[reagent]
        self.volume_ul -= taken
        return moved

    def evaporate(self, fraction: float) -> None:
        """Lose solvent, keep solute. This is what concentrates a sample."""
        self.volume_ul = max(0.0, self.volume_ul * (1.0 - fraction))


class Plate:
    """A microplate, and everything that has happened to it."""

    def __init__(
        self,
        barcode: str,
        *,
        plate_format: str = "96",
        label: str = "",
        lidded: bool = True,
    ) -> None:
        if plate_format not in FORMATS:
            raise ValueError(f"unknown plate format {plate_format!r}; known: {sorted(FORMATS)}")
        self.barcode = barcode
        self.format = plate_format
        self.label = label or barcode
        self.lidded = lidded
        rows, columns, working, dead = FORMATS[plate_format]
        self.rows, self.columns = rows, columns
        self.working_volume_ul = working
        self.dead_volume_ul = dead
        self.wells: dict[str, Well] = {
            well_name(r, c): Well() for r in range(rows) for c in range(columns)
        }
        self.temperature_c = 22.0
        self.created = time.time()
        self.last_touched = time.time()
        #: Where the plate physically is: a device id, or "bench".
        self.location = "bench"

    @property
    def well_count(self) -> int:
        return self.rows * self.columns

    def well(self, name: str) -> Well:
        key = self.normalise(name)
        if key not in self.wells:
            raise ValueError(
                f"{name!r} is not a well on a {self.format}-well plate "
                f"(A1 to {well_name(self.rows - 1, self.columns - 1)})"
            )
        return self.wells[key]

    def normalise(self, name: str) -> str:
        row, column = parse_well(name)
        return well_name(row, column)

    def occupied(self) -> dict[str, Well]:
        return {k: w for k, w in self.wells.items() if w.volume_ul > 0}

    def total_volume_ul(self) -> float:
        return sum(w.volume_ul for w in self.wells.values())

    def age_s(self) -> float:
        return time.time() - self.created

    def apply_evaporation(self, seconds: float, temperature_c: float) -> float:
        """Evaporate as a function of temperature and exposure.

        Roughly Arrhenius-shaped and calibrated so an unlidded 96-well plate at
        37 C loses a few percent an hour, which is the number that actually
        bites: a plate left uncovered overnight in an incubator comes back
        concentrated, and an agent that ignores that gets wrong potencies.
        """
        if seconds <= 0:
            return 0.0
        base_per_hour = 0.004 if self.lidded else 0.030
        rate = base_per_hour * math.exp((temperature_c - 22.0) / 18.0)
        fraction = 1.0 - math.exp(-rate * seconds / 3600.0)
        for well in self.wells.values():
            if well.volume_ul > 0:
                well.evaporate(fraction)
        return fraction

    def summary(self) -> dict[str, Any]:
        used = self.occupied()
        return {
            "barcode": self.barcode,
            "label": self.label,
            "format": self.format,
            "wells": self.well_count,
            "occupied_wells": len(used),
            "total_volume_ul": round(self.total_volume_ul(), 3),
            "lidded": self.lidded,
            "location": self.location,
            "temperature_c": round(self.temperature_c, 2),
            "reagents": sorted({r for w in used.values() for r in w.contents}),
        }

    def well_table(self) -> list[dict[str, Any]]:
        """Every non-empty well, for an artifact or a tool result."""
        rows = []
        for name, well in sorted(self.occupied().items(), key=lambda kv: parse_well(kv[0])):
            row: dict[str, Any] = {"well": name, "volume_ul": round(well.volume_ul, 3)}
            for reagent, pmol in sorted(well.contents.items()):
                row[f"{reagent}_uM"] = round(well.concentration_um(reagent), 4)
            rows.append(row)
        return rows


class PlateStore:
    """Process-wide bench: the plates this simulated lab contains.

    A module-level singleton, which is normally a smell. Here it is the point:
    the instruments are separate objects precisely so they can disagree, and
    they must nonetheless be able to touch the same physical plate.
    """

    def __init__(self) -> None:
        self._plates: dict[str, Plate] = {}

    def create(
        self, barcode: str, *, plate_format: str = "96", label: str = "", lidded: bool = True
    ) -> Plate:
        if barcode in self._plates:
            raise ValueError(f"a plate with barcode {barcode!r} already exists")
        plate = Plate(barcode, plate_format=plate_format, label=label, lidded=lidded)
        self._plates[barcode] = plate
        return plate

    def get(self, barcode: str) -> Plate:
        try:
            return self._plates[barcode]
        except KeyError:
            raise KeyError(
                f"no plate with barcode {barcode!r}; known plates: {sorted(self._plates) or 'none'}"
            ) from None

    def get_or_create(self, barcode: str, **kwargs: Any) -> Plate:
        return self._plates.get(barcode) or self.create(barcode, **kwargs)

    def all(self) -> dict[str, Plate]:
        return dict(self._plates)

    def discard(self, barcode: str) -> None:
        self._plates.pop(barcode, None)

    def clear(self) -> None:
        """Reset the bench. Used by tests, never in a running lab."""
        self._plates.clear()


#: The bench.
BENCH = PlateStore()
