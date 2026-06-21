"""Рабочее место роли: сенсор и АЦП.

Слой сенсора получает на вход полный спектральный куб (H, W, bands) после
геометрической проекции камеры-обскуры (диафрагма ахроматична, см. TODO в
`optics/optics_transformer.py`). Физический тракт фотоматрицы:

    1. Спектральная чувствительность регистрирующего элемента — свёртка спектра
       по кривым чувствительности R/G/B фотодиодов под цветными фильтрами
       (`apply_spectral_sensitivity`). Так выполняется разделение по каналам.
    2. Матрица Байера (CFA) — каждый фотосайт регистрирует только один цвет в
       зависимости от своей позиции в мозаике (`apply_bayer_mosaic`). На выходе —
       одноканальный RAW-кадр.
    3. Накопление заряда и АЦП на RAW-мозаике (`integrate_sensor_charge`,
       `quantize_frame`).
    4. Демозаика методом ближайшего соседа — восстановление трёх каналов из
       мозаики Байера (`demosaic_nearest`).
"""

from __future__ import annotations

import math

from workspace.models import (
    AdcConfig,
    SensorConfig,
    SensorExposure,
    SpectralSensitivity,
)
from workspace.shared import integrate_sensor_charge, quantize_frame

# Шаблон фильтра Байера 2x2 (порядок: верх-лево, верх-право, низ-лево, низ-право).
# 'R'/'G'/'B' — цвет фильтра над соответствующим фотосайтом.
DEFAULT_BAYER_PATTERN = "RGGB"

_CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}


def build_default_sensor_config(exposure: SensorExposure) -> SensorConfig:
    """
    Возвращает дефолтную конфигурацию сенсора под размер входной экспозиции.
    """
    channel_irradiance = exposure.channel_irradiance
    if channel_irradiance is None:
        raise ValueError("SensorExposure.channel_irradiance отсутствует")

    height = len(channel_irradiance)
    width = len(channel_irradiance[0])

    return SensorConfig(
        resolution=(height, width),
        pixel_size_um=4.8,
        # Кривые чувствительности нормированы балансом белого (сумма каждой ≈ 1),
        # поэтому усиление поднято, чтобы RAW-сигнал занимал рабочий диапазон АЦП.
        gain=24000.0,
        dark_offset=0.002,
        quantum_efficiency=[1.0 for _ in exposure.spectral_axis.wave],
        description="Параметры фотоматрицы для роли sensor_adc",
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


def _gaussian(x: float, center: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - center) / sigma) ** 2)


def build_default_spectral_sensitivity(
    spectral_axis,
) -> SpectralSensitivity:
    """
    Строит дефолтные кривые спектральной чувствительности регистрирующего
    элемента — колоколообразные (гауссовы) отклики R/G/B каналов с пиками,
    типичными для кремниевой матрицы с фильтрами Байера:

        синий  ~ 460 нм, зелёный ~ 540 нм, красный ~ 600 нм.

    Кривые перекрываются (как у реальных CFA), что моделирует метамерию.

    Кривые проходят баланс белого: каждая нормируется на свою сумму по
    спектру, поэтому равноэнергетический («плоский») спектр даёт одинаковый
    отклик R=G=B (нейтрально-серый). Без этой нормировки зелёный канал, как
    самый широкий и центральный, доминировал бы и давал зелёно-голубой завал.
    """
    wave = list(spectral_axis.wave)

    red = [_gaussian(w, center=600.0, sigma=50.0) for w in wave]
    green = [_gaussian(w, center=540.0, sigma=45.0) for w in wave]
    blue = [_gaussian(w, center=460.0, sigma=40.0) for w in wave]

    # Баланс белого: нормировка каждой кривой на её интеграл по спектру.
    def _white_balance(curve: list[float]) -> list[float]:
        total = sum(curve)
        if total <= 0:
            return curve
        return [value / total for value in curve]

    return SpectralSensitivity(
        wave=wave,
        red=_white_balance(red),
        green=_white_balance(green),
        blue=_white_balance(blue),
    )


def apply_spectral_sensitivity(
    channel_irradiance: list[list[list[float]]],
    sensitivity: SpectralSensitivity,
    quantum_efficiency: list[float] | None = None,
) -> list[list[list[float]]]:
    """
    Свёртывает спектральный куб (H, W, bands) в RGB-отклик (H, W, 3) по кривым
    спектральной чувствительности регистрирующего элемента.

    Для каждого пикселя отклик канала — интеграл (сумма по спектральным полосам)
    произведения облучённости на чувствительность канала и квантовую эффективность:

        response_c = Σ_band  E[band] · sensitivity_c[band] · QE[band]
    """
    height = len(channel_irradiance)
    width = len(channel_irradiance[0])
    bands = len(sensitivity.wave)

    if quantum_efficiency is None:
        quantum_efficiency = [1.0] * bands

    sens = sensitivity.as_matrix()  # [band][channel]

    rgb_response: list[list[list[float]]] = []
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            spectrum = channel_irradiance[y][x]
            pixel = [0.0, 0.0, 0.0]
            for band in range(bands):
                energy = spectrum[band] * quantum_efficiency[band]
                pixel[0] += energy * sens[band][0]
                pixel[1] += energy * sens[band][1]
                pixel[2] += energy * sens[band][2]
            row.append(pixel)
        rgb_response.append(row)
    return rgb_response


def bayer_channel_at(y: int, x: int, pattern: str = DEFAULT_BAYER_PATTERN) -> int:
    """
    Возвращает индекс канала (0=R, 1=G, 2=B), который регистрирует фотосайт
    в позиции (y, x), согласно шаблону Байера 2x2.
    """
    color = pattern[(y % 2) * 2 + (x % 2)]
    return _CHANNEL_INDEX[color]


def apply_bayer_mosaic(
    rgb_response: list[list[list[float]]],
    pattern: str = DEFAULT_BAYER_PATTERN,
) -> list[list[float]]:
    """
    Накладывает цветной фильтр Байера: каждый фотосайт пропускает только свой
    цвет, поэтому из RGB-отклика (H, W, 3) остаётся одноканальная RAW-мозаика
    (H, W) — именно то, что физически измеряет матрица перед АЦП.
    """
    height = len(rgb_response)
    width = len(rgb_response[0])

    mosaic: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            channel = bayer_channel_at(y, x, pattern)
            row.append(rgb_response[y][x][channel])
        mosaic.append(row)
    return mosaic


def _nearest_offsets(pattern: str) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    Для каждой из 4 фаз мозаики (y%2, x%2) подбирает смещение (dy, dx) к
    ближайшему фотосайту каждого канала R/G/B. Шаблон периодичен с периодом 2,
    поэтому для Байера радиус поиска не превышает 1 пикселя.
    """
    offsets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for py in range(2):
        for px in range(2):
            per_channel: list[tuple[int, int]] = [(0, 0)] * 3
            for channel in range(3):
                best: tuple[int, int] | None = None
                best_dist = math.inf
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if bayer_channel_at(py + dy, px + dx, pattern) != channel:
                            continue
                        dist = abs(dy) + abs(dx)
                        if dist < best_dist:
                            best_dist = dist
                            best = (dy, dx)
                assert best is not None
                per_channel[channel] = best
            offsets[(py, px)] = per_channel
    return offsets


def demosaic_nearest(
    mosaic: list[list[int]] | list[list[float]],
    pattern: str = DEFAULT_BAYER_PATTERN,
) -> list[list[list[int]]]:
    """
    Демозаика методом ближайшего соседа: восстанавливает три канала (H, W, 3)
    из одноканальной мозаики Байера. Для каждого пикселя недостающие каналы
    берутся от ближайшего фотосайта соответствующего цвета (с зеркальным
    отражением на границах кадра). Это и есть разделение RAW-кадра по каналам.
    """
    height = len(mosaic)
    width = len(mosaic[0])
    offsets = _nearest_offsets(pattern)

    def clamp(value: int, limit: int) -> int:
        if value < 0:
            return -value  # отражение от границы
        if value >= limit:
            return 2 * limit - value - 2
        return value

    rgb: list[list[list[int]]] = []
    for y in range(height):
        row: list[list[int]] = []
        for x in range(width):
            phase = offsets[(y % 2, x % 2)]
            pixel: list[int] = []
            for channel in range(3):
                dy, dx = phase[channel]
                sy = clamp(y + dy, height)
                sx = clamp(x + dx, width)
                pixel.append(int(mosaic[sy][sx]))
            row.append(pixel)
        rgb.append(row)
    return rgb


def build_rgb_frame(
    exposure: SensorExposure,
    sensor_config: SensorConfig,
    adc_config: AdcConfig,
    sensitivity: SpectralSensitivity | None = None,
    bayer_pattern: str = DEFAULT_BAYER_PATTERN,
) -> list[list[list[int]]]:
    """
    Полный тракт сенсора: спектральный куб (H, W, bands) -> RGB-кадр (H, W, 3).

    Этапы:
        1. Спектральная чувствительность -> RGB-отклик (разделение по каналам).
        2. Мозаика Байера -> одноканальный RAW.
        3. Накопление заряда + АЦП на RAW-мозаике.
        4. Демозаика ближайшим соседом -> восстановленный RGB-кадр.
    """
    channel_irradiance = exposure.channel_irradiance
    if channel_irradiance is None:
        raise ValueError("SensorExposure.channel_irradiance отсутствует")

    if sensitivity is None:
        sensitivity = build_default_spectral_sensitivity(exposure.spectral_axis)

    # 1. Спектральная чувствительность регистрирующего элемента -> RGB-отклик.
    rgb_response = apply_spectral_sensitivity(
        channel_irradiance,
        sensitivity,
        quantum_efficiency=sensor_config.quantum_efficiency,
    )

    # 2. Фильтр Байера -> одноканальная RAW-мозаика.
    bayer_raw = apply_bayer_mosaic(rgb_response, pattern=bayer_pattern)

    # 3. Накопление заряда и квантование на RAW-мозаике.
    charge_map = integrate_sensor_charge(
        bayer_raw,
        gain=sensor_config.gain,
        dark_offset=sensor_config.dark_offset,
    )
    raw_frame = quantize_frame(
        charge_map,
        bit_depth=adc_config.bit_depth,
        full_scale=adc_config.full_scale,
    )

    # 4. Демозаика ближайшим соседом -> восстановление каналов RGB.
    return demosaic_nearest(raw_frame, pattern=bayer_pattern)
