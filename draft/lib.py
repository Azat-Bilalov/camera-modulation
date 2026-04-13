from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import struct


@dataclass
class SpectralAxis:
    wavelengths_nm: list[float]

    @property
    def band_count(self) -> int:
        return len(self.wavelengths_nm)


def create_uniform_axis(start_nm: int = 400, stop_nm: int = 700, step_nm: int = 20) -> SpectralAxis:
    wavelengths = list(range(start_nm, stop_nm + 1, step_nm))
    return SpectralAxis(wavelengths_nm=[float(value) for value in wavelengths])


def gaussian_spectrum(axis: SpectralAxis, center_nm: float, width_nm: float, amplitude: float = 1.0) -> list[float]:
    spectrum: list[float] = []
    for wavelength in axis.wavelengths_nm:
        exponent = -((wavelength - center_nm) ** 2) / (2.0 * width_nm**2)
        spectrum.append(amplitude * math.exp(exponent))
    return normalize_vector(spectrum)


def normalize_vector(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0.0:
        return [0.0 for _ in values]
    return [value / total for value in values]


def build_test_reflectance_map(height: int, width: int, axis: SpectralAxis) -> list[list[list[float]]]:
    band_count = axis.band_count
    background = [0.20 for _ in range(band_count)]
    red_patch = gaussian_spectrum(axis, center_nm=620.0, width_nm=40.0, amplitude=1.0)
    green_patch = gaussian_spectrum(axis, center_nm=540.0, width_nm=35.0, amplitude=1.0)

    data: list[list[list[float]]] = []
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            cell = background[:]
            if 4 <= y < height - 4 and 4 <= x < width - 4:
                mix = red_patch if x < width // 2 else green_patch
                cell = [0.15 + 0.85 * value for value in mix]
            row.append(cell)
        data.append(row)
    return data


def simulate_scene(
    axis: SpectralAxis,
    source_spectrum: list[float],
    reflectance_map: list[list[list[float]]],
) -> list[list[list[float]]]:
    height = len(reflectance_map)
    width = len(reflectance_map[0])
    scene: list[list[list[float]]] = []
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            pixel = [
                source_spectrum[band_index] * reflectance_map[y][x][band_index]
                for band_index in range(axis.band_count)
            ]
            row.append(pixel)
        scene.append(row)
    return scene


def spectral_energy_map(scene: list[list[list[float]]]) -> list[list[float]]:
    return [[sum(pixel) for pixel in row] for row in scene]


def apply_optical_encoder(scene: list[list[list[float]]], axis: SpectralAxis) -> dict[str, list[list[float]]]:
    split_index = axis.band_count // 2
    height = len(scene)
    width = len(scene[0])

    low_band: list[list[float]] = []
    high_band: list[list[float]] = []
    combined: list[list[float]] = []

    for y in range(height):
        low_row: list[float] = []
        high_row: list[float] = []
        combined_row: list[float] = []
        for x in range(width):
            spectrum = scene[y][x]
            mask_factor = 1.0 if (x + y) % 2 == 0 else 0.72
            low_value = sum(spectrum[:split_index]) * 0.95 * mask_factor
            high_value = sum(spectrum[split_index:]) * 0.90 * (2.0 - mask_factor)
            low_row.append(low_value)
            high_row.append(high_value)
            combined_row.append(low_value + high_value)
        low_band.append(low_row)
        high_band.append(high_row)
        combined.append(combined_row)

    return {
        "channel_low": low_band,
        "channel_high": high_band,
        "sensor_exposure": combined,
    }


def simulate_sensor(
    exposure_map: list[list[float]],
    gain: float = 1800.0,
    dark_offset: float = 0.002,
) -> list[list[float]]:
    charge: list[list[float]] = []
    for y, row in enumerate(exposure_map):
        charge_row: list[float] = []
        for x, value in enumerate(row):
            fixed_pattern = ((x * 13 + y * 7) % 11) / 5000.0
            charge_value = max(0.0, value * gain + dark_offset + fixed_pattern)
            charge_row.append(charge_value)
        charge.append(charge_row)
    return charge


def quantize_adc(charge_map: list[list[float]], bit_depth: int = 10, full_scale: float = 1023.0) -> list[list[int]]:
    max_code = (1 << bit_depth) - 1
    frame: list[list[int]] = []
    for row in charge_map:
        frame_row: list[int] = []
        for value in row:
            normalized = max(0.0, min(value / full_scale, 1.0))
            frame_row.append(int(round(normalized * max_code)))
        frame.append(frame_row)
    return frame


def normalize_frame_to_u8(frame: list[list[int]]) -> list[list[int]]:
    flat = [value for row in frame for value in row]
    minimum = min(flat)
    maximum = max(flat)
    if maximum == minimum:
        return [[0 for _ in row] for row in frame]

    image: list[list[int]] = []
    for row in frame:
        image_row: list[int] = []
        for value in row:
            scaled = int(round((value - minimum) * 255.0 / (maximum - minimum)))
            image_row.append(max(0, min(255, scaled)))
        image.append(image_row)
    return image


def save_grayscale_bmp(image: list[list[int]], path: Path) -> None:
    height = len(image)
    width = len(image[0])
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_bytes = bytearray()

    for row in reversed(image):
        for value in row:
            byte_value = max(0, min(255, value))
            pixel_bytes.extend((byte_value, byte_value, byte_value))
        pixel_bytes.extend(b"\x00" * row_padding)

    file_header_size = 14
    dib_header_size = 40
    pixel_data_offset = file_header_size + dib_header_size
    file_size = pixel_data_offset + len(pixel_bytes)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(b"BM")
        file.write(struct.pack("<I", file_size))
        file.write(struct.pack("<HH", 0, 0))
        file.write(struct.pack("<I", pixel_data_offset))

        file.write(struct.pack("<I", dib_header_size))
        file.write(struct.pack("<i", width))
        file.write(struct.pack("<i", height))
        file.write(struct.pack("<H", 1))
        file.write(struct.pack("<H", 24))
        file.write(struct.pack("<I", 0))
        file.write(struct.pack("<I", len(pixel_bytes)))
        file.write(struct.pack("<i", 2835))
        file.write(struct.pack("<i", 2835))
        file.write(struct.pack("<I", 0))
        file.write(struct.pack("<I", 0))
        file.write(pixel_bytes)


def summarize_matrix(matrix: list[list[float | int]]) -> str:
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    flat = [value for row in matrix for value in row]
    return (
        f"size={height}x{width}, "
        f"min={min(flat):.4f}, "
        f"max={max(flat):.4f}, "
        f"mean={sum(float(value) for value in flat) / len(flat):.4f}"
    )


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
