from __future__ import annotations

from typing import Any

from workspace.models import OpticalChannel, OpticsConfig, SensorExposure
from workspace.optics.optics_transformer import (
    build_optics_input,
    convert_scene_to_channels,
    convert_scene_to_sensor as convert_scene_to_sensor_model,
)


def build_default_optics_config() -> OpticsConfig:
    """
    Возвращает стандартную конфигурацию оптического кодера.

    Input:
    - входных параметров не требуется.

    Output:
    - `OpticsConfig` с двумя каналами, шахматной маской и суммированием каналов.

    Где смотреть пример:
    - `draft/03_optics/example.py`
    """

    return OpticsConfig(
        channel_count=2,
        split_strategy="low-high spectral split",
        mask_pattern="checkerboard amplitude mask",
        transmission_low=0.95,
        transmission_high=0.90,
        recombination_mode="sum",
        description="Заглушка конфигурации оптики для роли optics",
    )


def build_default_channels(scene: Any, optics_config: OpticsConfig) -> list[OpticalChannel]:
    """
    Формирует два промежуточных оптических канала из спектральной карты сцены.

    Input:
    - `scene`: `SpectralImage`, полученный от модуля `scene_source`.
    - `optics_config`: параметры разделения и передачи каналов.

    Output:
    - список `OpticalChannel`, который архитектор может логировать и сравнивать.

    Где смотреть пример:
    - `draft/03_optics/example.py`
    - `draft/lib.py` -> `apply_optical_encoder`
    """

    optical_data = convert_scene_to_channels(
        scene=scene,
        optics_config=optics_config,
    )
    return [
        OpticalChannel(
            channel_id="low",
            data=optical_data["channel_low"],
            transmission=optics_config.transmission_low,
            mask_id=optics_config.mask_pattern,
            prism_id="P1",
            description="Низкочастотный или коротковолновый канал заглушки",
        ),
        OpticalChannel(
            channel_id="high",
            data=optical_data["channel_high"],
            transmission=optics_config.transmission_high,
            mask_id=optics_config.mask_pattern,
            prism_id="P2",
            description="Высокочастотный или длинноволновый канал заглушки",
        ),
    ]


def convert_scene_to_sensor(scene: Any, optics_config: OpticsConfig) -> SensorExposure:
    """
    Преобразует сцену в итоговую экспозицию сенсора.

    Input:
    - `scene`: спектральная карта сцены.
    - `optics_config`: конфигурация оптики.

    Output:
    - `SensorExposure`, согласованный с модулем `sensor_adc`.
    """

    return convert_scene_to_sensor_model(scene=scene, optics_config=optics_config)


def build_default_exposure(scene: Any, optics_config: OpticsConfig) -> SensorExposure:
    """
    Формирует суммарную экспозицию на сенсоре после оптического тракта.

    Input:
    - `scene`: спектральная карта сцены.
    - `optics_config`: конфигурация оптики.

    Output:
    - `SensorExposure`, ожидаемый модулем `sensor_adc`.

    Где смотреть пример:
    - `draft/03_optics/example.py`
    - `draft/07_full_pipeline/main.py`
    """

    return convert_scene_to_sensor_model(scene=scene, optics_config=optics_config)


def build_exposure_from_scene_source(
    axis: Any,
    source: Any,
    scene_object: Any,
    optics_config: OpticsConfig,
) -> SensorExposure:
    """
    Собирает экспозицию напрямую из структур `workspace/scene_source`.

    Функция ожидает duck-typed объекты с полями:
    - `axis.wave` и `axis.bands_count`;
    - `source.spectrum`;
    - `scene_object.reflectance`.
    """

    optics_input = build_optics_input(axis=axis, source=source, scene_object=scene_object)
    return convert_scene_to_sensor_model(scene=optics_input.spectra, axis=optics_input.axis, optics_config=optics_config)
