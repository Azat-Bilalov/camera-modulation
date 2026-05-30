"""
Общие модели данных для всех ролевых подпапок `workspace/`.

Сейчас этот слой синхронизирован с черновым набором моделей из `draft/models.py`.
Архитектор использует этот файл как единый импортный вход для всех участников.
"""
from dataclasses import dataclass
from typing import list

from draft.models import (  # noqa: F401
    AdcConfig,
    ChargeMatrix,
    DigitalFrame,
    ExportConfig,
    ObjectConfig,
    OpticalChannel,
    OpticsConfig,
    PipelineArtifacts,
    ReconstructionConfig,
    SensorConfig,
    SensorExposure,
    SourceConfig,
    SpectralAxis,
    SpectralImage,
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
