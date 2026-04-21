from __future__ import annotations

from workspace.models import OpticalChannel, OpticsConfig, SensorExposure, SpectralImage
from workspace.shared import split_optical_channels


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


def build_default_channels(scene: SpectralImage, optics_config: OpticsConfig) -> list[OpticalChannel]:
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

    optical_data = split_optical_channels(scene.data, scene.spectral_axis)
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


def build_default_exposure(scene: SpectralImage, optics_config: OpticsConfig) -> SensorExposure:
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

    _ = optics_config
    optical_data = split_optical_channels(scene.data, scene.spectral_axis)
    return SensorExposure(
        irradiance=optical_data["sensor_exposure"],
        exposure_time_s=0.01,
        spectral_axis=scene.spectral_axis,
        description="Заглушка экспозиции на матрице после optics",
    )
