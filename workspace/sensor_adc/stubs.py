from __future__ import annotations
import numpy as np

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

    # Сформируем разрешение матрицы, для этого переведем список в numpy массив и посмотрим его shape
    data = np.array(exposure.channel_irradiance)
    height, width, _ = data.shape

    # exposure.channel_irradiance
    # print(exposure)
    return SensorConfig(
        resolution=(height, width),
        pixel_size_um=4.8,
        gain=2000.0,
        dark_offset=0.002,
        quantum_efficiency=[1.0 for _ in exposure.spectral_axis.wave],
        description="Заглушка параметров сенсора для роли sensor_adc",
    )
def build_bayer_matrix(sensor_config: SensorConfig) -> np.ndarray:
    """
    Функция построения матрицы Байера (CFA). Применяет шаблонную матрицу 2х2 на разрешение матрицы

    | G | R |
    | B | G |

    Input:
    :param sensor_config: Конфигурация сенсора под размер входной экспозиции.
    :return: Возвращает матрицу Байера в виде двумерного целочисленного массива,
     где следующие значения соответствуют определенным фильтрам:
     - R - 1;
     - G - 2;
     - B - 3;
    """
    # resolution хранится как (height, width) — см. build_default_sensor_config.
    height, width = sensor_config.resolution
    pattern = np.array([[2, 1], [3, 2]])
    tiles_y = (height + 1) // 2
    tiles_x = (width + 1) // 2
    # Замостим шаблоном и обрежем точно под разрешение (на случай нечётных размеров).
    return np.tile(pattern, (tiles_y, tiles_x))[:height, :width]


def apply_bayer_matrix_to_wavelength(
    exposure: SensorExposure, sensor_config: SensorConfig
) -> np.ndarray:
    """
    Применяет матрицу Байера (CFA) к карте освещённости каналов.

    Реальный пиксель сенсора стоит под одним цветным фильтром, поэтому из
    `(H, W, 3)` RGB-освещённости остаётся одно значение на пиксель — тот канал,
    который соответствует позиции пикселя в шаблоне Байера. Это и есть «сырой»
    (raw) mosaic-кадр, который дальше копит заряд.

    Input:
    - `exposure`: экспозиция с полем `channel_irradiance` формы `(H, W, 3)`.
    - `sensor_config`: конфигурация сенсора (нужна для шаблона Байера).

    Output:
    - двумерный `np.ndarray` формы `(H, W)` — raw mosaic.
    """

    data = np.array(exposure.channel_irradiance, dtype=float)  # (H, W, 3)
    bayer = build_bayer_matrix(sensor_config)                  # (H, W): 1=R, 2=G, 3=B
    # Коды Байера 1/2/3 -> индексы каналов 0/1/2 (R/G/B).
    channel_index = bayer - 1
    raw = np.take_along_axis(data, channel_index[:, :, None], axis=2)[:, :, 0]
    return raw


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

    # Сначала применяем CFA: (H, W, 3) -> raw mosaic (H, W) c одним каналом на пиксель.
    raw_mosaic = apply_bayer_matrix_to_wavelength(exposure, sensor_config)
    charge_map = integrate_sensor_charge(
        exposure_map=raw_mosaic.tolist(),
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
