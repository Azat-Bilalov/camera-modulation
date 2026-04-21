from __future__ import annotations

from workspace.models import AdcConfig, ChargeMatrix, DigitalFrame, SensorConfig, SensorExposure
from workspace.shared import integrate_sensor_charge, quantize_frame


def build_default_sensor_config(exposure: SensorExposure) -> SensorConfig:
    """
    Возвращает дефолтную конфигурацию сенсора под размер входной экспозиции.

    Input:
    - `exposure`: `SensorExposure`, пришедший от модуля `optics`.

    Output:
    - `SensorConfig` с размером, согласованным с картой `exposure.irradiance`.

    Где смотреть пример:
    - `draft/04_sensor/example.py`
    """

    resolution = (exposure.height, exposure.width)
    return SensorConfig(
        resolution=resolution,
        pixel_size_um=4.8,
        gain=2000.0,
        dark_offset=0.002,
        quantum_efficiency=[1.0 for _ in exposure.spectral_axis.wavelengths_nm],
        description="Заглушка параметров сенсора для роли sensor_adc",
    )


def build_default_charge(exposure: SensorExposure, sensor_config: SensorConfig) -> ChargeMatrix:
    """
    Преобразует экспозицию в карту заряда на матрице.

    Input:
    - `exposure`: карта энергии на сенсоре.
    - `sensor_config`: параметры усиления и темнового сигнала.

    Output:
    - `ChargeMatrix`, ожидаемый блоком АЦП.

    Где смотреть пример:
    - `draft/04_sensor/example.py`
    """

    charge_map = integrate_sensor_charge(
        exposure_map=exposure.irradiance,
        gain=sensor_config.gain,
        dark_offset=sensor_config.dark_offset,
    )
    return ChargeMatrix(
        charge=charge_map,
        sensor_config=sensor_config,
        description="Заглушка накопленного заряда для передачи в ADC",
    )


def build_default_adc_config() -> AdcConfig:
    """
    Возвращает дефолтную конфигурацию АЦП.

    Input:
    - входных параметров не требуется.

    Output:
    - `AdcConfig` с 10-битным квантованием и полным масштабом 900.

    Где смотреть пример:
    - `draft/05_adc/example.py`
    - `draft/07_full_pipeline/main.py`
    """

    return AdcConfig(
        bit_depth=10,
        full_scale=900.0,
        reference_voltage_v=3.3,
        amplification=1.0,
        saturation_mode="clip",
        description="Заглушка параметров АЦП для роли sensor_adc",
    )


def build_default_frame(charge: ChargeMatrix, adc_config: AdcConfig) -> DigitalFrame:
    """
    Квантует карту заряда в цифровой кадр.

    Input:
    - `charge`: `ChargeMatrix` от модели сенсора.
    - `adc_config`: параметры цифрового квантования.

    Output:
    - `DigitalFrame`, ожидаемый модулем визуализации и экспорта.

    Где смотреть пример:
    - `draft/05_adc/example.py`
    """

    data = quantize_frame(
        charge_map=charge.charge,
        bit_depth=adc_config.bit_depth,
        full_scale=adc_config.full_scale,
    )
    return DigitalFrame(
        data=data,
        bit_depth=adc_config.bit_depth,
        description="Заглушка цифрового кадра после квантования",
    )
