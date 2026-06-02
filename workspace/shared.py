from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path


def integrate_sensor_charge(
    exposure_map: list[list[float]],
    gain: float,
    dark_offset: float,
) -> list[list[float]]:
    """Переводит карту экспозиции в карту накопленного заряда."""

    charge: list[list[float]] = []
    for y, row in enumerate(exposure_map):
        charge_row: list[float] = []
        for x, value in enumerate(row):
            fixed_pattern = ((x * 13 + y * 7) % 11) / 5000.0
            charge_value = max(0.0, value * gain + dark_offset + fixed_pattern)
            charge_row.append(charge_value)
        charge.append(charge_row)
    return charge


def quantize_frame(
    charge_map: list[list[float]], bit_depth: int, full_scale: float
) -> list[list[int]]:
    """Квантование аналогового сигнала в целочисленный цифровой кадр."""

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
    """Нормализует цифровой кадр в диапазон 0..255."""

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
    """Сохраняет двумерный grayscale preview в 24-битный `bmp`."""

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


def normalize_rgb_to_u8(
    image: list[list[list[int]]], gamma: float = 0.4
) -> list[list[list[int]]]:
    """
    Нормализует RGB-кадр в диапазон 0..255 единым масштабом по всем каналам,
    чтобы не разрушить цветовой баланс, с опциональной гамма-коррекцией.

    Вместо min-max stretching используется деление на глобальный максимум
    (без вычитания минимума). Это сохраняет хроматичность даже в ярких
    пикселях после геометрического затухания 1/r².
    """

    flat = [value for row in image for pixel in row for value in pixel]
    maximum = max(flat)
    if maximum == 0:
        return [[[0, 0, 0] for _ in row] for row in image]

    result: list[list[list[int]]] = []
    for row in image:
        result_row: list[list[int]] = []
        for pixel in row:
            # Масштабирование относительно глобального максимума
            normed = [value / maximum for value in pixel]
            # Гамма-коррекция (одинаковая для всех каналов — сохраняет оттенок)
            corrected = [
                max(0, min(255, int(round((v**gamma) * 255.0)))) for v in normed
            ]
            result_row.append(corrected)
        result.append(result_row)
    return result


def save_rgb_bmp(image: list[list[list[int]]], path: Path) -> None:
    """Сохраняет RGB-кадр `(H, W, 3)` со значениями 0..255 в 24-битный `bmp`."""

    height = len(image)
    width = len(image[0])
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_bytes = bytearray()

    for row in reversed(image):
        for pixel in row:
            red, green, blue = pixel
            # BMP хранит пиксели в порядке B, G, R.
            pixel_bytes.extend(
                (
                    max(0, min(255, blue)),
                    max(0, min(255, green)),
                    max(0, min(255, red)),
                )
            )
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


def summarize_matrix(matrix: Sequence[Sequence[float | int]]) -> str:
    """Возвращает короткую строку со статистикой по двумерной матрице."""

    height = len(matrix)
    width = len(matrix[0]) if height else 0
    flat = [value for row in matrix for value in row]
    return (
        f"size={height}x{width}, "
        f"min={min(flat):.4f}, "
        f"max={max(flat):.4f}, "
        f"mean={sum(float(value) for value in flat) / len(flat):.4f}"
    )
