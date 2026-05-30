"""Рабочее место роли: модель сцены и источника.
Гостев М.А."""

from dataclasses import dataclass
from typing import List
import re
from pathlib import Path
import math


@dataclass
class SpectralAxis:
    wave: List[float]
    start: float
    stop: float
    bands_count: int

@dataclass
class SpectralImage:
    spectral_axis: SpectralAxis
    data: List[List[List[float]]] 

@dataclass
class SourceConfig:
    spectrum: List[float]
    position: List[float]

@dataclass
class ObjectConfig:
    reflectance: List[float]
    width: int
    height: int
    point_size: float

def read_spectrum_from_txt(file_path: str) -> List[float]:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    first_line = text.strip().splitlines()[0]

    values = re.findall(r"[-+]?\d*\.\d+|\d+", first_line)
    return [float(value) for value in values]


def build_axis_by_step(start: float, step: float, count: int) -> SpectralAxis:
    wave = [start + i * step for i in range(count)]
    return SpectralAxis(
        wave=wave,
        start=wave[0],
        stop=wave[-1],
        bands_count=len(wave)
    )

def calculate_distance(point: List[float], source_position: List[float]) -> float:
    dx = source_position[0] - point[0]
    dy = source_position[1] - point[1]
    dz = source_position[2] - point[2]

    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def calculate_cos_angle(point: List[float], source_position: List[float]) -> float:
    r = calculate_distance(point, source_position)

    if r == 0:
        raise ValueError("Источник не может находиться точно в точке объекта.")

    dz = source_position[2] - point[2]
    cos_angle = dz / r

    return max(cos_angle, 0.0)

def build_optic_input(
    axis: SpectralAxis,
    source: SourceConfig,
    obj: ObjectConfig
) -> SpectralImage:

    if len(source.spectrum) != axis.bands_count:
        raise ValueError("Количество значений спектра не совпадает с количеством каналов.")

    if len(obj.reflectance) != axis.bands_count:
        raise ValueError("Количество коэффициентов отражения не совпадает с количеством каналов.")
    
    spectral_data = []

    for y in range(obj.height):
        row = []

        for x in range(obj.width):
            point_position = [
                x * obj.point_size,
                y * obj.point_size,
                0.0
            ]

            r = calculate_distance(point_position, source.position)
            cos_angle = calculate_cos_angle(point_position, source.position)

            point_spectrum = [
                round(
                    source.spectrum[band] * cos_angle / (r ** 2) * obj.reflectance[band],
                    4
                )
                for band in range(axis.bands_count)
            ]

            row.append(point_spectrum)

        spectral_data.append(row)

    return SpectralImage(
        spectral_axis=axis,
        data=spectral_data
    )
