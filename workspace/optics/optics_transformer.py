from dataclasses import dataclass
from typing import List
from workspace.models import SpectralImage, SpectralAxis, SensorExposure, OpticsConfig

# ------------------- Генераторы недостающих данных -------------------
def generate_split_wavelength(axis: SpectralAxis) -> float:
    """Возвращает границу разбиения спектра на два канала (нм)."""
    return 550.0  # общепринятая граница видимого диапазона

def generate_transmittance(wavelengths_nm: List[float]) -> List[float]:
    """
    Генерирует коэффициенты пропускания оптической системы.
    Простейшая линейная модель: растёт от 0.8 до 1.0 в видимом диапазоне.
    """
    wl_min = 400.0
    wl_max = 700.0
    transmittance = []
    for wl in wavelengths_nm:
        if wl < wl_min:
            t = 0.8
        elif wl > wl_max:
            t = 1.0
        else:
            t = 0.8 + 0.2 * (wl - wl_min) / (wl_max - wl_min)
        transmittance.append(t)
    return transmittance

def generate_coding_masks(height: int, width: int):
    """
    Создаёт две комплементарные кодирующие маски.
    mask_A: шахматный узор (1 на чётных суммах координат, 0 иначе)
    mask_B: инвертированная mask_A.
    """
    mask_A = [[0.0]*width for _ in range(height)]
    mask_B = [[0.0]*width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if (x + y) % 2 == 0:
                mask_A[y][x] = 1.0
                mask_B[y][x] = 0.0
            else:
                mask_A[y][x] = 0.0
                mask_B[y][x] = 1.0
    return mask_A, mask_B

def convert_scene_to_sensor(
    scene: SpectralImage,
    optics_config: OpticsConfig
) -> SensorExposure:
    """
    Преобразует SpectralImage в SensorExposure согласно конфигурации OpticsConfig.

    Поддерживаемые режимы:
    - split_mode = 'low_high' : разделение спектра на два канала по
      пороговой длине волны split_threshold_nm. Канал A: λ ≤ порог,
      канал B: λ > порог.
    - mask_pattern = 'checkerboard' : комплементарные шахматные маски.
    - transmission_low / transmission_high : линейное пропускание от 400 до 700 нм.
    - recombination_mode = 'sum' : суммирование вкладов каналов с масками.
    Время экспозиции фиксировано = 1.0 с.
    """
    if optics_config.channel_count != 2:
        raise NotImplementedError(
            f"Поддерживается только channel_count=2, передано {optics_config.channel_count}"
        )

    axis = scene.spectral_axis
    wl = axis.wavelengths_nm
    height = scene.height
    width = scene.width

    # --- 1. Разбиение спектра с помощью low-high spectral split ---
    if optics_config.split_strategy == "low-high spectral split":
        split_wl = optics_config.split_threshold_nm
        # Канал A: низкие (low) длины волн, ≤ порога
        indices_A = [i for i, w in enumerate(wl) if w <= split_wl]
        # Канал B: высокие (high) длины волн, > порога
        indices_B = [i for i, w in enumerate(wl) if w > split_wl]
    else:
        raise ValueError(
            f"Неизвестная стратегия разделения: '{optics_config.split_strategy}'. "
            f"Допустима только 'low_high'."
        )

    # --- 2. Кодирующие маски ---
    if optics_config.mask_pattern == "checkerboard amplitude mask":
        mask_A = [[0.0] * width for _ in range(height)]
        mask_B = [[0.0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                if (x + y) % 2 == 0:
                    mask_A[y][x] = 1.0
                    mask_B[y][x] = 0.0
                else:
                    mask_A[y][x] = 0.0
                    mask_B[y][x] = 1.0
    elif optics_config.mask_pattern == "none":
        # Без масок: везде единицы, суммируем оба канала
        mask_A = [[1.0] * width for _ in range(height)]
        mask_B = [[1.0] * width for _ in range(height)]
    else:
        raise ValueError(
            f"Неизвестный шаблон маски: '{optics_config.mask_pattern}'. "
            f"Допустимы 'checkerboard', 'none'."
        )

    # --- 3. Коэффициенты пропускания (линейная модель) ---
    def compute_transmittance(wavelength):
        if wavelength <= 400.0:
            return optics_config.transmission_low
        if wavelength >= 700.0:
            return optics_config.transmission_high
        t = (wavelength - 400.0) / 300.0
        return optics_config.transmission_low + t * (optics_config.transmission_high - optics_config.transmission_low)

    transmittance = [compute_transmittance(w) for w in wl]

    # --- 4. Интегрирование и суммирование на сенсоре ---
    def integrate_channel(pixel_spectrum, band_indices):
        """Энергия в заданном наборе спектральных каналов с учётом пропускания."""
        if len(band_indices) < 2:
            if not band_indices:
                return 0.0
            idx = band_indices[0]
            if 0 < idx < len(wl) - 1:
                dw = (wl[idx] - wl[idx-1]) / 2.0 + (wl[idx+1] - wl[idx]) / 2.0
            elif idx > 0:
                dw = (wl[idx] - wl[idx-1]) / 2.0
            else:
                dw = (wl[1] - wl[0]) / 2.0 if len(wl) > 1 else 1.0
            return pixel_spectrum[idx] * transmittance[idx] * dw

        integral = 0.0
        for i in range(len(band_indices) - 1):
            idx1 = band_indices[i]
            idx2 = band_indices[i+1]
            avg_val = (pixel_spectrum[idx1] * transmittance[idx1] +
                       pixel_spectrum[idx2] * transmittance[idx2]) / 2.0
            dw = wl[idx2] - wl[idx1]
            integral += avg_val * dw
        return integral

    irradiance = [[0.0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            spectrum = scene.data[y][x]
            energy_A = integrate_channel(spectrum, indices_A)
            energy_B = integrate_channel(spectrum, indices_B)

            if optics_config.recombination_mode == "sum":
                irradiance[y][x] = mask_A[y][x] * energy_A + mask_B[y][x] * energy_B
            else:
                raise ValueError(
                    f"Неизвестный режим рекомбинации: '{optics_config.recombination_mode}'. "
                    f"Допустим только 'sum'."
                )

    exposure_time = 1.0
    description = (
        f"Low-high split на {split_wl:.1f} нм, "
        f"маски '{optics_config.mask_pattern}', "
        f"пропускание {optics_config.transmission_low:.2f}–{optics_config.transmission_high:.2f}, "
        f"рекомбинация '{optics_config.recombination_mode}', "
        f"время экспозиции {exposure_time} с."
    )

    return SensorExposure(
        irradiance=irradiance,
        exposure_time_s=exposure_time,
        spectral_axis=axis,
        description=description
    )