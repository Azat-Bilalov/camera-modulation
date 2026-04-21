"""
Общие модели данных для всех ролевых подпапок `workspace/`.

Сейчас этот слой синхронизирован с черновым набором моделей из `draft/models.py`.
Архитектор использует этот файл как единый импортный вход для всех участников.
"""

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
