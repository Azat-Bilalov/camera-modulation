"""Рабочее место роли: модель сцены и источника.
Гостев М.А."""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from workspace.models import (
    ObjectConfig,
    SourceConfig,
    SpectralAxis,
    SpectralImage,
)


@dataclass
class SceneSourceInput:
    radiation: list[float]  # спектр излучения источника, Вт/м²/нм
    source_xyz: list[float]  # координаты X, Y, Z в метрах
    reflectance: list[float]  # коэффициенты отражения по спектральным каналам, доли
    object_width: int  # пиксели
    object_height: int  # пиксели
    point_size: float  # метры на пиксель
    power: float = 1.0  # мощность источника, Вт
    tilt_deg: float = 0.0  # угол наклона источника к нормали поверхности, градусы

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


def read_spectrum_from_csv(file_path: str, column: str = "value") -> list[float]:
    """
    Читает спектральные значения из CSV-файла.

    По умолчанию ожидает колонку `value` (как в `sample_spectrum.csv`).
    """
    values: list[float] = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def read_source_spectrum_from_csv(file_path: str, column: str = "value") -> list[float]:
    """
    Читает спектр излучения источника из CSV (Вт/м²/нм).

    По умолчанию ожидает колонку `value`, как в текущих входных CSV проекта.
    """
    values: list[float] = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


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


def apply_tilt(cos_angle: float, tilt_deg: float) -> float:
    """Корректирует косинус с учётом угла наклона источника, градусы -> радианы."""

    tilt_rad = math.radians(tilt_deg)
    return cos_angle * math.cos(tilt_rad)


def build_optic_input(
    axis: SpectralAxis,
    source: SourceConfig,
    obj: ObjectConfig,
    power: float = 1.0,
    tilt_deg: float = 0.0,
) -> SpectralImage:
    """Строит спектральное поле сцены с учётом мощности источника и угла наклона."""

    if len(source.spectrum) != axis.bands_count:
        raise ValueError("Количество значений спектра не совпадает с количеством каналов.")

    if len(obj.reflectance) != axis.bands_count:
        raise ValueError("Количество коэффициентов отражения не совпадает с количеством каналов.")

    spectral_data = []

    for y in range(obj.height):
        row = []

        for x in range(obj.width):
            point_position = [x * obj.point_size, y * obj.point_size, 0.0]

            r = calculate_distance(point_position, source.position)
            cos_angle = calculate_cos_angle(point_position, source.position)
            # Коррекция косинуса на наклон источника в градусах.
            cos_corrected = apply_tilt(cos_angle, tilt_deg)

            point_spectrum = [
                power * source.spectrum[band] * cos_corrected / (r**2) * obj.reflectance[band]
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
        power=input_data.power,
        tilt_deg=input_data.tilt_deg,
    )

    return SceneSourceArtifacts(
        axis=axis,
        source=source,
        object_config=object_config,
        scene=scene,
    )


def get_scene_source_input(
    reflectance_csv: str | None = None,
    source_csv: str | None = None,
    power: float = 1.0,
    tilt_deg: float = 0.0,
) -> SceneSourceInput:
    """
    Формирует входные данные для сцены.

    Если передан `reflectance_csv`, значения `reflectance` читаются из CSV
    (колонка `value`). В противном случае используется дефолтный путь
    ``workspace/input/sample_spectrum.csv``.

    Если передан `source_csv`, ``radiation`` читается из CSV по колонке
    `value`. Иначе задаётся единичным спектром той же длины
    (равномерное освещение по всем длинам волн).
    """
    if reflectance_csv is None:
        reflectance_csv = str(Path(__file__).resolve().parents[1] / "input" / "sample_spectrum.csv")

    spectrum = read_spectrum_from_csv(reflectance_csv)
    max_val = max(spectrum)
    reflectance = [v / max_val for v in spectrum]

    if source_csv is not None:
        radiation = read_source_spectrum_from_csv(source_csv)
    else:
        radiation = [1.0] * len(spectrum)

    return SceneSourceInput(
        radiation=radiation,
        source_xyz=[10.0, 10.0, 50.0],
        reflectance=reflectance,
        object_height=32,
        object_width=32,
        point_size=10,
        power=power,
        tilt_deg=tilt_deg,
    )
