"""Рабочее место роли: модель сцены и источника.
Гостев М.А."""

from dataclasses import dataclass, field
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
    spectrum: List[float]          # radiation

@dataclass
class ObjectConfig:
    reflectance: List[float]       # coef


# Тестовые данные
testW = [400, 420, 440, 460, 480, 500]
testR = [0.2, 0.35, 0.6, 0.8, 0.7, 0.5]
testC = [0.1, 0.12, 0.18, 0.3, 0.45, 0.5]


# Преобразование входных данных в список
def parse_list(text):
    values = text.replace(";", ",").split(",")
    result = []

    for value in values:
        value = value.strip()
        if value:
            result.append(float(value))
    return result


# Формирование данных для модуля оптики
def build_optic_input(axis: SpectralAxis, source: SourceConfig, obj: ObjectConfig) -> SceneSignal:
    signal = [round(source.spectrum[i] * obj.reflectance[i], 4) for i in range(axis.bands_count)]
    return SceneSignal(spectral_axis=axis, input_signal=signal)

def build_axis(wave: List[float]) -> SpectralAxis:
    return SpectralAxis(wave=wave, start=wave[0], stop=wave[-1], bands_count=len(wave))

if input("Использовать тестовые данные? (y/n): ").lower() == "y":
    wave = testW
    radiation = testR
    coef = testC
else:
    wave = parse_list(input("Длины волн, нм: "))
    radiation = parse_list(input("Мощность излучения: "))
    coef = parse_list(input("Коэффициенты отражения объекта: "))

if __name__ == "__main__":
    spectral_axis = build_axis(wave)
    source_config = SourceConfig(spectrum=radiation)
    object_config = ObjectConfig(reflectance=coef)
    optic_input = build_optic_input(spectral_axis, source_config, object_config)


print(f"Начальная длина волны: {optic_input.spectral_axis.start} нм")
print(f"Конечная длина волны: {optic_input.spectral_axis.stop} нм")
print(f"Количество спектральных каналов: {optic_input.spectral_axis.bands_count}")
print(f"Длины волн: {optic_input.spectral_axis.wave}")
print(f"Спектральная карта сцены: {optic_input.input_signal}")
print(f"Данные для модуля оптики: {optic_input}")