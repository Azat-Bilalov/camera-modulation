"""Рабочее место роли: оптический тракт.
Андронаки Е.Д.

Камера-обскура: точечная диафрагма без линзы. Каждая точка объекта посылает
луч через диафрагму на плоскость изображения, формируя перевёрнутую проекцию.
Энергия луча ослабляется по радиометрической формуле камеры-обскуры
(закон cos^4 — естественное виньетирование).

Диафрагма ахроматична (геометрическая оптика, без призмы/дисперсии), поэтому
разбиение спектра на R/G/B-каналы сюда не относится — оно выполняется в слое
сенсора по спектральной чувствительности регистрирующего элемента и фильтру
Байера (см. `sensor_adc/sensor_pipeline.py`). Здесь на выход отдаётся полный
спектральный куб (H, W, bands) после геометрической проекции.
"""

from __future__ import annotations

import math

from workspace.models import OpticsConfig, SensorExposure, SpectralAxis, SpectralImage


def convert_scene_to_exposure(
    scene: SpectralImage,
    config: OpticsConfig,
) -> SensorExposure:
    optical_data = project_scene_through_pinhole(scene, config, scene.spectral_axis)
    return SensorExposure(channel_irradiance=optical_data, spectral_axis=scene.spectral_axis)


def build_default_optics_config() -> OpticsConfig:
    return OpticsConfig(
        channel_count=3,
        split_strategy="pinhole projection (camera obscura)",
        mask_pattern="single circular aperture",
        transmission=[0.95, 0.90, 0.85],  # R, G, B — синий обычно пропускается хуже
        recombination_mode="multi-channel",
        rgb_ranges_nm=[(380, 480), (480, 600), (600, 730)],
        aperture_diameter=50.0,
        object_distance=50.0,
        image_distance=50.0,
        aperture_offset_x=0.0,
        aperture_offset_y=0.0,
        tilt_x_deg=0.0,
        tilt_y_deg=0.0,
        description="Камера-обскура: проекция сцены через точечную диафрагму",
    )


Vector3 = tuple[float, float, float]


def _camera_basis(tilt_x_deg: float, tilt_y_deg: float) -> tuple[Vector3, Vector3, Vector3]:
    """
    Строит ортонормированный базис камеры (оптическая ось, «право», «верх»)
    с учётом наклона диафрагмы/сенсора вокруг осей X (`tilt_x_deg`) и
    Y (`tilt_y_deg`). При нулевых углах ось совпадает с мировой +Z, как
    в исходной (не наклонённой) камере-обскуре.
    """
    tx = math.radians(tilt_x_deg)
    ty = math.radians(tilt_y_deg)

    # Базовая ось (0, 0, 1), повёрнутая сначала вокруг X, затем вокруг Y.
    axis = (
        math.cos(tx) * math.sin(ty),
        -math.sin(tx),
        math.cos(tx) * math.cos(ty),
    )

    # "Право" = world_up x axis, нормированный.
    right = (axis[2], 0.0, -axis[0])
    right_norm = math.hypot(right[0], right[2])
    if right_norm < 1e-9:
        # Ось камеры почти совпала с мировым "верхом" — берём произвольный базис.
        right = (1.0, 0.0, 0.0)
        right_norm = 1.0
    right = (right[0] / right_norm, right[1] / right_norm, right[2] / right_norm)

    # "Верх" = axis x right (оба единичные и ортогональные -> результат единичный).
    up = (
        axis[1] * right[2] - axis[2] * right[1],
        axis[2] * right[0] - axis[0] * right[2],
        axis[0] * right[1] - axis[1] * right[0],
    )

    return axis, right, up


def _project_point_3d(
    point: Vector3,
    camera_position: Vector3,
    axis: Vector3,
    right: Vector3,
    up: Vector3,
    image_distance: float,
) -> tuple[float, float, float] | None:
    """
    Перспективная (центральная) проекция точки объекта через диафрагму на
    плоскость изображения, заданную базисом (`axis`, `right`, `up`).

    Возвращает координаты точки на плоскости изображения `(u_img, w_img)`
    относительно центра сенсора и косинус угла `alpha` между лучом и
    оптической осью камеры (для расчёта виньетирования). Возвращает `None`,
    если точка находится позади диафрагмы — её невозможно отобразить.
    """
    v = (point[0] - camera_position[0], point[1] - camera_position[1], point[2] - camera_position[2])

    d_axial = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2]
    if d_axial <= 0:
        return None

    u = v[0] * right[0] + v[1] * right[1] + v[2] * right[2]
    w = v[0] * up[0] + v[1] * up[1] + v[2] * up[2]

    r_lateral = math.hypot(u, w)
    cos_alpha = d_axial / math.hypot(d_axial, r_lateral)

    # Инверсия изображения: луч, продолженный через диафрагму, меняет знак.
    scale = image_distance / d_axial
    u_img = -scale * u
    w_img = -scale * w

    return u_img, w_img, cos_alpha


def _vignetting_factor(cos_alpha: float, aperture_diameter: float, image_distance: float) -> float:
    """
    Радиометрическая формула камеры-обскуры (закон cos^4, естественное
    виньетирование точечной диафрагмы):

        E_image = L_object * (pi / 4) * (D / v)^2 * cos^4(alpha)

    где `L_object` — энергетическая яркость точки объекта, `D` — диаметр
    диафрагмы, `v` — расстояние «диафрагма -> плоскость изображения»,
    `alpha` — угол между лучом «точка объекта -> диафрагма» и оптической
    осью камеры. Чем дальше точка от оптической оси, тем больше угол и
    тем сильнее затухание — это даёт характерное затемнение к краям кадра
    (при смещении/наклоне камеры — несимметричное).
    """
    return (math.pi / 4.0) * (aperture_diameter / image_distance) ** 2 * cos_alpha**4


def project_scene_through_pinhole(
    scene: SpectralImage | list[list[list[float]]],
    optics_config: OpticsConfig,
    axis: SpectralAxis | None = None,
) -> list[list[list[float]]]:
    """
    Камера-обскура: сцена проецируется через точечную диафрагму на плоскость
    изображения (с инверсией по X и Y), мощность каждой точки ослабляется по
    радиометрической формуле камеры-обскуры (`_vignetting_factor`).

    Спектр на выходе НЕ разбивается на R/G/B — диафрагма ахроматична, поэтому
    разбиение по длинам волн выполняется на слое сенсора (см. описание в начале
    файла и в `sensor_adc/sensor_pipeline.py`).
    """
    if isinstance(scene, SpectralImage):
        scene_data = scene.data
        spectral_axis = scene.spectral_axis
    else:
        if axis is None:
            raise ValueError("Параметр axis обязателен.")
        scene_data = scene
        spectral_axis = axis

    bands_count = spectral_axis.bands_count
    height = len(scene_data)
    width = len(scene_data[0])

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # Диафрагма может быть смещена относительно центра сцены и наклонена —
    # это даёт перспективные искажения и несимметричное виньетирование.
    camera_position: Vector3 = (
        cx + optics_config.aperture_offset_x,
        cy + optics_config.aperture_offset_y,
        -optics_config.object_distance,
    )
    cam_axis, cam_right, cam_up = _camera_basis(optics_config.tilt_x_deg, optics_config.tilt_y_deg)

    sensor_exposure: list[list[list[float]]] = [[[0.0] * bands_count for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            spectrum = scene_data[y][x]

            projection = _project_point_3d(
                point=(float(x), float(y), 0.0),
                camera_position=camera_position,
                axis=cam_axis,
                right=cam_right,
                up=cam_up,
                image_distance=optics_config.image_distance,
            )
            if projection is None:
                continue  # точка позади диафрагмы — не отображается
            u_img, w_img, cos_alpha = projection

            # Энергия точки объекта, дошедшая через диафрагму до плоскости изображения.
            factor = _vignetting_factor(
                cos_alpha=cos_alpha,
                aperture_diameter=optics_config.aperture_diameter,
                image_distance=optics_config.image_distance,
            )

            # Перевёрнутая (инвертированная) проекция точки на плоскость изображения.
            xi, yi = round(cx + u_img), round(cy + w_img)

            if 0 <= xi < width and 0 <= yi < height:
                pixel = sensor_exposure[yi][xi]
                for band in range(bands_count):
                    pixel[band] += spectrum[band] * factor
            # Иначе точка проецируется за пределы кадра сенсора и отбрасывается
            # (виньетирование по полю кадра камеры-обскуры).

    return sensor_exposure  # 3D: H×W×bands (свёртка в H×W×3 — в слое сенсора)
