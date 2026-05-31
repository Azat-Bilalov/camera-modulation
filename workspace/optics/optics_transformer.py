from workspace.models import SpectralImage, OpticsConfig,SpectralAxis, SensorExposure



def build_default_optics_config() -> OpticsConfig:
    return OpticsConfig(
        channel_count=3,
        split_strategy="rgb spectral split",
        mask_pattern="checkerboard amplitude mask",
        transmission=[0.95, 0.90, 0.85],  # R, G, B — синий обычно пропускается хуже
        recombination_mode="multi-channel",
        rgb_ranges_nm=[(400, 500), (500, 600), (600, 700)],
        description="RGB конфигурация оптики",
    )

def convert_scene_to_exposure(
    scene: SpectralImage,
    config: OpticsConfig,
) -> SensorExposure:
    optical_data = convert_scene_to_channels_rgb(scene, config, scene.spectral_axis)
    return SensorExposure(
        channel_irradiance=optical_data,
        spectral_axis=scene.spectral_axis
    )
    
def build_default_optics_config() -> OpticsConfig:
    return OpticsConfig(
        channel_count=3,
        split_strategy="rgb spectral split",
        mask_pattern="checkerboard amplitude mask",
        transmission=[0.95, 0.90, 0.85],  # R, G, B — синий обычно пропускается хуже
        recombination_mode="multi-channel",
        rgb_ranges_nm=[(400, 500), (500, 600), (600, 700)],
        description="RGB конфигурация оптики",
    )

def convert_scene_to_channels_rgb(
    scene: SpectralImage | list[list[list[float]]],
    optics_config: OpticsConfig,
    axis: SpectralAxis | None = None,
) -> list[list[list[float]]]:
    """
    RGB-версия: разделяет спектр на 3 канала по длинам волн.
    """
    if isinstance(scene, SpectralImage):
        scene_data = scene.data
        spectral_axis = scene.spectral_axis
    else:
        if axis is None:
            raise ValueError("Параметр axis обязателен.")
        scene_data = scene
        spectral_axis = axis

    wavelengths = spectral_axis.wave
    
    # Определяем, какие индексы спектра попадают в R, G, B
    r_indices = [i for i, w in enumerate(wavelengths) if 600 <= w <= 730]
    g_indices = [i for i, w in enumerate(wavelengths) if 500 <= w < 600]
    b_indices = [i for i, w in enumerate(wavelengths) if 380 <= w < 500]
    
    height = len(scene_data)
    width = len(scene_data[0])

    r_band: list[list[float]] = []
    g_band: list[list[float]] = []
    b_band: list[list[float]] = []
    sensor_exposure: list[list[list[float]]] = []  # (H, W, 3)

    for y in range(height):
        r_row: list[float] = []
        g_row: list[float] = []
        b_row: list[float] = []
        exposure_row: list[list[float]] = []
        for x in range(width):
            spectrum = scene_data[y][x]

            # Суммируем энергию в каждой спектральной полосе
            r_value = sum(spectrum[i] for i in r_indices) * optics_config.transmission[0]
            g_value = sum(spectrum[i] for i in g_indices) * optics_config.transmission[1]
            b_value = sum(spectrum[i] for i in b_indices) * optics_config.transmission[2]

            r_row.append(r_value)
            g_row.append(g_value)
            b_row.append(b_value)
            exposure_row.append([r_value, g_value, b_value])

        r_band.append(r_row)
        g_band.append(g_row)
        b_band.append(b_row)
        sensor_exposure.append(exposure_row)

    return sensor_exposure  # теперь 3D: H×W×3