from workspace.scene_source import SpectralAxis, SpectralImage, SensorExposure

def convert_scene_to_exposure(
    image: SpectralImage
) -> list[list[dict[str, list[float]]]]:
    blue_border = 490
    green_border = 570
    red_border = 730
    blue_interval_steps = ((blue_border - image.spectral_axis.start) / 10) + 1
    green_interval_steps = ((green_border - blue_border) / 10) + 1
    red_interval_steps = ((min(red_border, image.spectral_axis.stop) - green_border) / 10) + 1

    blue_list = list()
    green_list = list()
    red_list = list()

    result = list()

    for i in range(len(image.data)):
        column = list()
        for j in range(len(image.data[i])):
            row = list()
            for k in range(len(image.data[i][j])):
                if k < blue_interval_steps:
                    blue_list.append(image.data[i][j][k])
                elif blue_interval_steps <= k < green_interval_steps:
                    green_list.append(image.data[i][j][k])
                else:
                    red_list.append(image.data[i][j][k])
            row.append({"blue": blue_list, "green": green_list, "red": red_list})
        column.append(row)
    
    result
            
    
    return list(list(dict()))


    

def convert_scene_to_channels(
    scene: SpectralImage | list[list[list[float]]],
    optics_config: OpticsConfig,
    axis: SpectralAxis | None = None,
) -> dict[str, list[list[float]]]:
    """
    Преобразует спектральную карту сцены в оптические каналы и карту экспозиции.

    Input:
    - `scene`: `SpectralImage` или спектральная карта `(H, W, bands)`.
    - `optics_config`: параметры оптического разделения и передачи.
    - `axis`: опциональная ось, нужна только если `scene` передан как матрица.

    Output:
    - словарь с ключами `channel_low`, `channel_high`, `sensor_exposure`.

    Примечание:
    - В текущей версии используются те же детерминированные маски, что и
      в `split_optical_channels`, но коэффициенты передачи теперь берутся
      напрямую из `OpticsConfig`.
    """

    if isinstance(scene, SpectralImage):
        scene_data = scene.data
        spectral_axis = scene.spectral_axis
    else:
        if axis is None:
            raise ValueError("Параметр axis обязателен, когда scene передан как матрица.")
        scene_data = scene
        spectral_axis = axis

    split_index = spectral_axis.band_count // 2
    height = len(scene_data)
    width = len(scene_data[0])

    low_band: list[list[float]] = []
    high_band: list[list[float]] = []
    sensor_exposure: list[list[float]] = []

    for y in range(height):
        low_row: list[float] = []
        high_row: list[float] = []
        exposure_row: list[float] = []
        for x in range(width):
            spectrum = scene_data[y][x]
            mask_factor = 1.0 if (x + y) % 2 == 0 else 0.72

            low_value = sum(spectrum[:split_index]) * optics_config.transmission_low * mask_factor
            high_value = (
                sum(spectrum[split_index:])
                * optics_config.transmission_high
                * (2.0 - mask_factor)
            )

            low_row.append(low_value)
            high_row.append(high_value)
            exposure_row.append(low_value + high_value)

        low_band.append(low_row)
        high_band.append(high_row)
        sensor_exposure.append(exposure_row)

    return {
        "channel_low": low_band,
        "channel_high": high_band,
        "sensor_exposure": sensor_exposure,
    }


def convert_scene_to_sensor(
    scene: SpectralImage | list[list[list[float]]],
    optics_config: OpticsConfig,
    axis: SpectralAxis | None = None,
    exposure_time_s: float = 0.01,
) -> SensorExposure:
    """
    Преобразует сцену в итоговый объект `SensorExposure`.

    Input:
    - `scene`: `SpectralImage` или спектральная карта `(H, W, bands)`.
    - `optics_config`: параметры оптического разделения и передачи.
    - `axis`: опциональная ось, нужна только если `scene` передан как матрица.
    - `exposure_time_s`: время экспозиции в секундах.

    Output:
    - `SensorExposure` для модуля `sensor_adc`.
    """

    if isinstance(scene, SpectralImage):
        spectral_axis = scene.spectral_axis
    else:
        if axis is None:
            raise ValueError("Параметр axis обязателен, когда scene передан как матрица.")
        spectral_axis = axis

    optical_data = convert_scene_to_channels(
        scene=scene,
        axis=axis,
        optics_config=optics_config,
    )
    return SensorExposure(
        irradiance=optical_data["sensor_exposure"],
        exposure_time_s=exposure_time_s,
        spectral_axis=spectral_axis,
        description="Экспозиция на матрице после преобразования сцены в sensor plane",
    )