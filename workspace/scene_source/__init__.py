"""Рабочее место роли: модель сцены и источника.
Гостев М.А."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SpectralAxis:
    wave: List[float]
    start: float
    stop: float
    bands_count: int


@dataclass
class SceneSignal:
    spectral_axis: SpectralAxis
    input_signal: List[float]


@dataclass
class SourceConfig:
    spectrum: List[float]


@dataclass
class ObjectConfig:
    reflectance: List[float]


def read_spectrum_from_txt(file_path: str) -> List[float]:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    first_line = text.strip().splitlines()[0]

    values = re.findall(r"[-+]?\d*\.\d+|\d+", first_line)
    return [float(value) for value in values]


def build_axis_by_step(start: float, step: float, count: int) -> SpectralAxis:
    wave = [start + i * step for i in range(count)]
    return SpectralAxis(wave=wave, start=wave[0], stop=wave[-1], bands_count=len(wave))


def build_optic_input(
    axis: SpectralAxis, source: SourceConfig, obj: ObjectConfig
) -> SceneSignal:

    if len(source.spectrum) != axis.bands_count:
        raise ValueError(
            "Количество значений спектра не совпадает с количеством каналов."
        )

    if len(obj.reflectance) != axis.bands_count:
        raise ValueError(
            "Количество коэффициентов отражения не совпадает с количеством каналов."
        )

    signal = [
        round(source.spectrum[i] * obj.reflectance[i], 4)
        for i in range(axis.bands_count)
    ]

    return SceneSignal(spectral_axis=axis, input_signal=signal)


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "greenMeasure.txt"

    radiation = read_spectrum_from_txt(file_path)

    spectral_axis = build_axis_by_step(start=380.0, step=10.0, count=len(radiation))

    # Пока отражение считаем единичным
    coef = [1.0 for _ in radiation]

    source_config = SourceConfig(spectrum=radiation)
    object_config = ObjectConfig(reflectance=coef)

    optic_input = build_optic_input(spectral_axis, source_config, object_config)

    print(f"Начальная длина волны: {optic_input.spectral_axis.start} нм")
    print(f"Конечная длина волны: {optic_input.spectral_axis.stop} нм")
    print(f"Количество спектральных каналов: {optic_input.spectral_axis.bands_count}")
    print(f"Длины волн: {optic_input.spectral_axis.wave}")
    print(f"Спектральная карта сцены: {optic_input.input_signal}")
    print(f"Данные для модуля оптики: {optic_input}")
