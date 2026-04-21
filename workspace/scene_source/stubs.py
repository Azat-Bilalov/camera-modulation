from __future__ import annotations

from workspace.models import ObjectConfig, SourceConfig, SpectralAxis, SpectralImage
from workspace.shared import (
    build_default_reflectance_map,
    create_default_axis,
    gaussian_spectrum,
    simulate_scene_matrix,
)


def build_default_axis() -> SpectralAxis:
    """
    Возвращает общую спектральную ось для модуля сцены и источника.

    Input:
    - входных параметров не требуется.

    Output:
    - `SpectralAxis` с дефолтной сеткой 400..700 нм с шагом 20 нм.

    Где смотреть пример:
    - `draft/01_spectral_axis/example.py`
    """

    return create_default_axis(start_nm=400, stop_nm=700, step_nm=20)


def build_default_source(axis: SpectralAxis) -> SourceConfig:
    """
    Строит тестовый источник света в общей модели данных.

    Input:
    - `axis`: спектральная ось, на которой должен быть задан источник.

    Output:
    - `SourceConfig` с гауссовым спектром и направлением вдоль оси камеры.

    Где смотреть пример:
    - `draft/02_scene/example.py`
    - `draft/07_full_pipeline/main.py`
    """

    spectrum = gaussian_spectrum(axis, center_nm=560.0, width_nm=90.0, amplitude=1.0)
    return SourceConfig(
        source_type="synthetic-gaussian",
        intensity=1.0,
        spectrum=spectrum,
        description="Заглушка источника света для роли scene_source",
    )


def build_default_object(axis: SpectralAxis, height: int = 32, width: int = 40) -> ObjectConfig:
    """
    Строит тестовый объект с картой отражения, совместимой с draft-примерами.

    Input:
    - `axis`: спектральная ось.
    - `height`, `width`: размер рабочей сцены.

    Output:
    - `ObjectConfig` с `reflectance_map` формата `(H, W, bands)`.

    Где смотреть пример:
    - `draft/lib.py` -> `build_test_reflectance_map`
    - `draft/02_scene/example.py`
    """

    reflectance_map = build_default_reflectance_map(height=height, width=width, axis=axis)
    return ObjectConfig(
        object_name="test-target",
        height=height,
        width=width,
        reflectance_map=reflectance_map,
        description="Заглушка сцены: фон + красная и зеленая области",
    )


def build_default_scene(
    axis: SpectralAxis,
    source: SourceConfig,
    object_config: ObjectConfig,
) -> SpectralImage:
    """
    Собирает спектральную карту сцены из источника и объекта.

    Input:
    - `axis`: общая спектральная ось.
    - `source`: источник света с уже рассчитанным спектром.
    - `object_config`: объект с картой отражения.

    Output:
    - `SpectralImage`, ожидаемый модулем оптики.

    Где смотреть пример:
    - `draft/02_scene/example.py`
    """

    data = simulate_scene_matrix(
        axis=axis,
        source_spectrum=source.spectrum,
        reflectance_map=object_config.reflectance_map,
    )
    return SpectralImage(
        data=data,
        spectral_axis=axis,
        source_name=source.source_type,
        description="Заглушка спектральной карты сцены для передачи в optics",
    )
