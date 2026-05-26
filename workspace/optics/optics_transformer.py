from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from workspace.models import OpticsConfig, SensorExposure, SpectralAxis, SpectralImage

from workspace.scene_source import SceneSignal


SpectralCube = list[list[list[float]]]


@dataclass(frozen=True)
class OpticsInput:
    wavelengths_nm: list[float]
    spectra: SpectralCube

    @property
    def axis(self) -> SpectralAxis:
        return SpectralAxis(wavelengths_nm=self.wavelengths_nm)


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _read_axis(axis_like: Any) -> list[float]:
    if axis_like is None:
        raise ValueError("Axis is required for raw optics input.")

    if hasattr(axis_like, "wave"):
        wavelengths = _plain(getattr(axis_like, "wave"))
        expected_count = getattr(axis_like, "bands_count", len(wavelengths))
    elif hasattr(axis_like, "wavelengths_nm"):
        wavelengths = _plain(getattr(axis_like, "wavelengths_nm"))
        expected_count = getattr(axis_like, "band_count", len(wavelengths))
    else:
        raise TypeError("Axis must expose either wave or wavelengths_nm.")

    if not _is_sequence(wavelengths):
        raise TypeError("Axis wavelengths must be a sequence.")

    result = [float(value) for value in wavelengths]
    if not result:
        raise ValueError("Axis must contain at least one wavelength.")
    if int(expected_count) != len(result):
        raise ValueError(f"Axis band count is {expected_count}, but {len(result)} wavelengths were given.")
    return result


def _read_vector(value: Any, label: str) -> list[float]:
    value = _plain(value)
    if not _is_sequence(value):
        raise TypeError(f"{label} must be a numeric sequence.")
    if not all(isinstance(item, Real) for item in value):
        raise TypeError(f"{label} must contain only numbers.")
    return [float(item) for item in value]


def _is_vector(value: Any) -> bool:
    value = _plain(value)
    return _is_sequence(value) and all(isinstance(item, Real) for item in value)


def _cube_from_vector(signal: list[float]) -> SpectralCube:
    return [[signal]]


def _read_cube(value: Any) -> SpectralCube:
    value = _plain(value)
    if not _is_sequence(value):
        raise TypeError("Scene data must be a cube with shape (height, width, bands).")

    cube: SpectralCube = []
    for row_index, row in enumerate(value):
        row = _plain(row)
        if not _is_sequence(row):
            raise TypeError(f"Scene row {row_index} is not a sequence.")

        cube_row: list[list[float]] = []
        for column_index, spectrum in enumerate(row):
            cube_row.append(_read_vector(spectrum, f"scene spectrum ({row_index}, {column_index})"))
        cube.append(cube_row)
    return cube


def _validate_input(optics_input: OpticsInput) -> None:
    bands = len(optics_input.wavelengths_nm)
    if not optics_input.spectra:
        raise ValueError("Scene must contain at least one row.")
    if not optics_input.spectra[0]:
        raise ValueError("Scene must contain at least one pixel.")

    width = len(optics_input.spectra[0])
    for row_index, row in enumerate(optics_input.spectra):
        if len(row) != width:
            raise ValueError(f"Scene row {row_index} has width {len(row)} instead of {width}.")
        for column_index, spectrum in enumerate(row):
            if len(spectrum) != bands:
                raise ValueError(
                    f"Scene spectrum ({row_index}, {column_index}) has {len(spectrum)} bands; "
                    f"expected {bands}."
                )


def build_optics_input(
    axis: Any,
    source: Any,
    scene_object: Any,
) -> OpticsInput:
    """Build optics input from scene_source-style axis, SourceConfig, and ObjectConfig."""

    wavelengths = _read_axis(axis)
    source_spectrum = _read_vector(getattr(source, "spectrum"), "source spectrum")
    reflectance = _read_vector(getattr(scene_object, "reflectance"), "object reflectance")

    if len(source_spectrum) != len(wavelengths):
        raise ValueError("Source spectrum length does not match axis band count.")
    if len(reflectance) != len(wavelengths):
        raise ValueError("Object reflectance length does not match axis band count.")

    signal = [source_spectrum[index] * reflectance[index] for index in range(len(wavelengths))]
    optics_input = OpticsInput(wavelengths_nm=wavelengths, spectra=_cube_from_vector(signal))
    _validate_input(optics_input)
    return optics_input


def read_optics_input(scene: Any, axis: Any | None = None) -> OpticsInput:
    """Read SceneSignal-like input from workspace/scene_source without importing that package."""

    wavelengths = _read_axis(axis)
    optics_input = OpticsInput(wavelengths_nm=wavelengths, spectra=scene)

    return optics_input


def _split_indices(wavelengths: list[float]) -> tuple[list[int], list[int]]:
    if len(wavelengths) == 1:
        return [0], []

    cutoff_nm = (min(wavelengths) + max(wavelengths)) / 2.0
    low = [index for index, wavelength in enumerate(wavelengths) if wavelength <= cutoff_nm]
    high = [index for index, wavelength in enumerate(wavelengths) if wavelength > cutoff_nm]

    if not low or not high:
        split_index = max(1, len(wavelengths) // 2)
        low = list(range(split_index))
        high = list(range(split_index, len(wavelengths)))
    return low, high


def _mask_pair(x: int, y: int, optics_config: OpticsConfig) -> tuple[float, float]:
    pattern = optics_config.mask_pattern.lower()
    if "checker" not in pattern:
        return 1.0, 1.0

    low_mask = 1.0 if (x + y) % 2 == 0 else 0.72
    high_mask = 2.0 - low_mask
    return low_mask, high_mask


def _combine(low_value: float, high_value: float, optics_config: OpticsConfig) -> float:
    mode = optics_config.recombination_mode.lower()
    if mode == "sum":
        return low_value + high_value
    if mode in {"mean", "average"}:
        return (low_value + high_value) / 2.0
    raise ValueError(f"Unsupported recombination mode: {optics_config.recombination_mode}")


def convert_scene_to_channels(
    scene: SceneSignal,
    optics_config: OpticsConfig,
    axis: Any | None = None,
) -> dict[str, list[list[float]]]:
    optics_input = read_optics_input(scene, axis=axis)
    low_indices, high_indices = _split_indices(optics_input.wavelengths_nm)

    low_channel: list[list[float]] = []
    high_channel: list[list[float]] = []
    sensor_exposure: list[list[float]] = []

    for y, row in enumerate(optics_input.spectra):
        low_row: list[float] = []
        high_row: list[float] = []
        exposure_row: list[float] = []

        for x, spectrum in enumerate(row):
            low_mask, high_mask = _mask_pair(x, y, optics_config)
            low_value = sum(spectrum[index] for index in low_indices) * optics_config.transmission_low * low_mask
            high_value = sum(spectrum[index] for index in high_indices) * optics_config.transmission_high * high_mask

            low_row.append(low_value)
            high_row.append(high_value)
            exposure_row.append(_combine(low_value, high_value, optics_config))

        low_channel.append(low_row)
        high_channel.append(high_row)
        sensor_exposure.append(exposure_row)

    return {
        "channel_low": low_channel,
        "channel_high": high_channel,
        "sensor_exposure": sensor_exposure,
    }


def convert_scene_to_sensor(
    scene: Any,
    optics_config: OpticsConfig,
    axis: Any | None = None,
    exposure_time_s: float = 0.01,
) -> SensorExposure:
    optics_input = read_optics_input(scene, axis=axis)
    optical_data = convert_scene_to_channels(optics_input.spectra, optics_config=optics_config, axis=optics_input.axis)

    return SensorExposure(
        irradiance=optical_data["sensor_exposure"],
        exposure_time_s=exposure_time_s,
        spectral_axis=optics_input.axis,
        description="Sensor exposure produced by the optics stage from scene_source input.",
    )
