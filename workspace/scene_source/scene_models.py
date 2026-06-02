"""Рабочее место роли: модель сцены и источника.
Гостев М.А."""

import math
import re
from dataclasses import dataclass

from workspace.models import (
    ObjectConfig,
    SourceConfig,
    SpectralAxis,
    SpectralImage,
)


@dataclass
class SceneSourceInput:
    radiation: list[float]
    source_xyz: list[float]
    reflectance: list[float]
    object_width: int
    object_height: int
    point_size: float


@dataclass
class SceneSourceArtifacts:
    axis: SpectralAxis
    source: SourceConfig
    object_config: ObjectConfig
    scene: SpectralImage


def read_spectrum_from_txt(file_path: str) -> list[float]:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    first_line = text.strip().splitlines()[0]

    values = re.findall(r"[-+]?\d*\.\d+|\d+", first_line)
    return [float(value) for value in values]


def build_axis_by_step(start: float, step: float, count: int) -> SpectralAxis:
    wave = [start + i * step for i in range(count)]
    return SpectralAxis(wave=wave, start=wave[0], stop=wave[-1], bands_count=len(wave))


def calculate_distance(point: list[float], source_position: list[float]) -> float:
    dx = source_position[0] - point[0]
    dy = source_position[1] - point[1]
    dz = source_position[2] - point[2]

    return math.sqrt(dx**2 + dy**2 + dz**2)


def calculate_cos_angle(point: list[float], source_position: list[float]) -> float:
    r = calculate_distance(point, source_position)

    if r == 0:
        raise ValueError("Источник не может находиться точно в точке объекта.")

    dz = source_position[2] - point[2]
    cos_angle = dz / r

    return max(cos_angle, 0.0)


def build_optic_input(
    axis: SpectralAxis, source: SourceConfig, obj: ObjectConfig
) -> SpectralImage:

    if len(source.spectrum) != axis.bands_count:
        raise ValueError(
            "Количество значений спектра не совпадает с количеством каналов."
        )

    if len(obj.reflectance) != axis.bands_count:
        raise ValueError(
            "Количество коэффициентов отражения не совпадает с количеством каналов."
        )

    spectral_data = []

    for y in range(obj.height):
        row = []

        for x in range(obj.width):
            point_position = [x * obj.point_size, y * obj.point_size, 0.0]

            r = calculate_distance(point_position, source.position)
            cos_angle = calculate_cos_angle(point_position, source.position)

            point_spectrum = [
                round(
                    source.spectrum[band] * cos_angle / (r**2) * obj.reflectance[band],
                    4,
                )
                for band in range(axis.bands_count)
            ]

            row.append(point_spectrum)

        spectral_data.append(row)

    return SpectralImage(spectral_axis=axis, data=spectral_data)


def build_scene_source(input_data: SceneSourceInput) -> SceneSourceArtifacts:
    axis = build_axis_by_step(
        start=380.0,
        step=10.0,
        count=len(input_data.radiation),
    )

    source = SourceConfig(
        spectrum=input_data.radiation,
        position=input_data.source_xyz,
    )

    object_config = ObjectConfig(
        reflectance=input_data.reflectance,
        width=input_data.object_width,
        height=input_data.object_height,
        point_size=input_data.point_size,
    )

    scene = build_optic_input(
        axis=axis,
        source=source,
        obj=object_config,
    )

    return SceneSourceArtifacts(
        axis=axis,
        source=source,
        object_config=object_config,
        scene=scene,
    )


def get_scene_source_input() -> SceneSourceInput:
    ref = [
        7.19772,
        8.66835,
        10.0241,
        10.6711,
        10.9908,
        11.318,
        11.713,
        12.13,
        12.5267,
        12.8175,
        13.015,
        13.1402,
        13.2771,
        13.4178,
        13.5352,
        13.54,
        13.4372,
        13.1751,
        12.6869,
        12.1096,
        11.3222,
        10.3386,
        9.22227,
        8.32678,
        7.86045,
        7.65831,
        7.57516,
        7.56342,
        7.72733,
        8.0386,
        8.40108,
        8.71172,
        8.92745,
        9.01649,
        9.01843,
        9.165,
    ]
    return SceneSourceInput(
        radiation=[1] * len(ref),
        source_xyz=[10.0, 10.0, 50.0],
        reflectance=ref,
        object_height=32,
        object_width=32,
        point_size=10,
    )
