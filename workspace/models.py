"""
Общие модели данных для всех ролевых подпапок `workspace/`.

Сейчас этот слой синхронизирован с черновым набором моделей из `draft/models.py`.
Архитектор использует этот файл как единый импортный вход для всех участников.
"""
from dataclasses import dataclass

from draft.models import (  # noqa: F401
    AdcConfig,
    ChargeMatrix,
    DigitalFrame,
    ExportConfig,
    OpticalChannel,
    PipelineArtifacts,
    ReconstructionConfig,
    SensorConfig,
)

@dataclass
class SpectralAxis:
    wave: list[float]
    start: float
    stop: float
    bands_count: int


@dataclass
class SpectralImage:
    spectral_axis: SpectralAxis
    data: list[list[list[float]]]  # [y][x][band]


@dataclass
class SourceConfig:
    spectrum: list[float]
    position: list[float]  # [x, y, z]


@dataclass
class ObjectConfig:
    reflectance: list[float]
    width: int
    height: int
    point_size: float

@dataclass
class OpticsConfig:
    channel_count: int
    split_strategy: str
    mask_pattern: str
    transmission: list[float]  # [R, G, B] — вместо transmission_low/high
    recombination_mode: str = "sum"
    rgb_ranges_nm: list[tuple[float, float]] = None  # [(400,500), (500,600), (600,700)]
    description: str = "..."

@dataclass
class SensorExposure:
    spectral_axis: SpectralAxis
    channel_irradiance: list[list[list[float]]] | None = None  # (H, W, 3) — новое поле
    description: str = "Распределение энергии на матрице до электроники"
    exposure_time_s: float = 0.01

__all__ = [
    "AdcConfig",
    "ChargeMatrix",
    "DigitalFrame",
    "ExportConfig",
    "ObjectConfig",
    "OpticalChannel",
    "OpticsConfig",
    "PipelineArtifacts",
    "ReconstructionConfig",
    "SensorConfig",
    "SensorExposure",
    "SourceConfig",
    "SpectralAxis",
    "SpectralImage",
]
