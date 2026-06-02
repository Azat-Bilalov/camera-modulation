from __future__ import annotations

import numpy as np

from workspace.models import (
    AdcConfig,
    SensorConfig,
    SensorExposure,
)
from workspace.shared import integrate_sensor_charge, quantize_frame


def build_default_sensor_config(exposure: SensorExposure) -> SensorConfig:
    """
    Возвращает дефолтную конфигурацию сенсора под размер входной экспозиции.
    """
    data = np.array(exposure.channel_irradiance)
    height, width, _ = data.shape

    return SensorConfig(
        resolution=(height, width),
        pixel_size_um=4.8,
        gain=2000.0,
        dark_offset=0.002,
        quantum_efficiency=[1.0 for _ in exposure.spectral_axis.wave],
        description="Заглушка параметров сенсора для роли sensor_adc",
    )


def build_default_adc_config() -> AdcConfig:
    """
    Возвращает дефолтную конфигурацию АЦП.
    """
    return AdcConfig(
        bit_depth=10,
        full_scale=8.0,
        reference_voltage_v=3.3,
        amplification=1.0,
        saturation_mode="clip",
        description="Заглушка параметров АЦП для роли sensor_adc",
    )


def build_rgb_frame(
    exposure: SensorExposure,
    sensor_config: SensorConfig,
    adc_config: AdcConfig,
) -> list[list[list[int]]]:
    """
    Преобразует 3-канальную экспозицию (H, W, 3) в 3-канальный цифровой кадр.

    Каждый канал проходит усиление и АЦП независимо (без Bayer CFA).
    """
    channel_irradiance = exposure.channel_irradiance
    if channel_irradiance is None:
        raise ValueError("SensorExposure.channel_irradiance отсутствует")

    height = len(channel_irradiance)
    width = len(channel_irradiance[0])

    channel_frames: list[list[list[int]]] = []
    for c in range(3):
        channel_2d = [
            [channel_irradiance[y][x][c] for x in range(width)] for y in range(height)
        ]
        charge_map = integrate_sensor_charge(
            channel_2d,
            gain=sensor_config.gain,
            dark_offset=sensor_config.dark_offset,
        )
        frame_2d = quantize_frame(
            charge_map,
            bit_depth=adc_config.bit_depth,
            full_scale=adc_config.full_scale,
        )
        channel_frames.append(frame_2d)

    # Собираем обратно в (H, W, 3)
    rgb_frame: list[list[list[int]]] = []
    for y in range(height):
        row: list[list[int]] = []
        for x in range(width):
            row.append([channel_frames[c][y][x] for c in range(3)])
        rgb_frame.append(row)

    return rgb_frame
