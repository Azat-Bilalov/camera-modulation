from __future__ import annotations

from pathlib import Path

from workspace.models import DigitalFrame, ExportConfig, PipelineArtifacts, ReconstructionConfig
from workspace.shared import normalize_frame_to_u8, save_grayscale_bmp, summarize_matrix


def build_default_reconstruction_config() -> ReconstructionConfig:
    """
    Возвращает дефолтные параметры постобработки цифрового кадра.

    Input:
    - входных параметров не требуется.

    Output:
    - `ReconstructionConfig`.

    Где смотреть пример:
    - `draft/06_reconstruction_export/example.py`
    """

    return ReconstructionConfig(
        normalize_to_u8=True,
        defect_correction=True,
        contrast_stretch=True,
        description="Заглушка параметров реконструкции для роли visualization",
    )


def build_default_export_config(output_dir: str | None = None) -> ExportConfig:
    """
    Возвращает настройки сохранения итогового кадра и служебных артефактов.

    Input:
    - `output_dir`: путь к каталогу вывода; если не задан, берется `workspace/outputs`.

    Output:
    - `ExportConfig`.

    Где смотреть пример:
    - `draft/06_reconstruction_export/example.py`
    - `draft/07_full_pipeline/main.py`
    """

    target_dir = output_dir or str(Path(__file__).resolve().parents[1] / "outputs")
    return ExportConfig(
        output_dir=target_dir,
        image_format="bmp",
        save_intermediate=True,
        report_name="workspace_report.txt",
        description="Заглушка параметров экспорта для роли visualization",
    )


def build_default_preview(frame: DigitalFrame, reconstruction_config: ReconstructionConfig) -> list[list[int]]:
    """
    Подготавливает простой preview-кадр для архитектора и для тестовой визуализации.

    Input:
    - `frame`: цифровой кадр после АЦП.
    - `reconstruction_config`: параметры постобработки.

    Output:
    - двумерная матрица `0..255`, пригодная для сохранения в `bmp`.

    Где смотреть пример:
    - `draft/06_reconstruction_export/example.py`
    """

    _ = reconstruction_config
    return normalize_frame_to_u8(frame.data)


def export_default_preview(image: list[list[int]], export_config: ExportConfig) -> Path:
    """
    Сохраняет preview-изображение в файл на основе `ExportConfig`.

    Input:
    - `image`: двумерный массив 0..255.
    - `export_config`: настройки каталога и формата вывода.

    Output:
    - `Path` к сохраненному изображению.

    Где смотреть пример:
    - `draft/07_full_pipeline/main.py`
    """

    output_dir = Path(export_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"final_preview.{export_config.image_format}"
    save_grayscale_bmp(image, image_path)
    return image_path


def build_default_report(artifacts: PipelineArtifacts, image_path: Path) -> str:
    """
    Собирает подробный текстовый отчет для архитектора и роли верификации.
    """
    # Импортируем функции верификации
    from workspace.visualization.verifier import (
        calculate_image_statistics,
        verify_digital_range,
        verify_no_clipping
    )

    # Получаем статистику по кадру
    frame_stats = calculate_image_statistics(artifacts.frame.data, artifacts.frame.bit_depth)
    range_check = verify_digital_range(artifacts.frame.data, artifacts.frame.bit_depth)
    clip_check = verify_no_clipping(artifacts.frame.data, artifacts.frame.bit_depth)

    # Формируем отчёт
    report_lines = [
        "=" * 60,
        "ОТЧЁТ ПО РАБОТЕ МОДУЛЯ ВИЗУАЛИЗАЦИИ И ЭКСПОРТА",
        "=" * 60,
        "",
        "--- ОСНОВНАЯ ИНФОРМАЦИЯ ---",
        f"Спектральных диапазонов: {artifacts.axis.band_count}",
        f"Оптические каналы: {[ch.channel_id for ch in artifacts.optical_channels]}",
        "",
        "--- ПАРАМЕТРЫ СЕНСОРА ---",
        f"Экспозиция: {summarize_matrix(artifacts.exposure.irradiance)}",
        f"Заряд: {summarize_matrix(artifacts.charge.charge)}",
        "",
        "--- ЦИФРОВОЙ КАДР ---",
        f"Размер: {frame_stats['height']}x{frame_stats['width']}",
        f"Битность: {frame_stats['bit_depth']} бит",
        f"Диапазон значений: {frame_stats['dynamic_range']}",
        f"Среднее значение: {frame_stats['mean']}",
        "",
        "--- ВЕРИФИКАЦИЯ ---",
        f"Диапазон корректен: {range_check['is_valid']}",
        f"Клиппинг (макс): {clip_check['clipped_high']} пикс. ({clip_check['clipped_high_percent']}%)",
        f"Клиппинг (мин): {clip_check['clipped_low']} пикс. ({clip_check['clipped_low_percent']}%)",
        f"Клиппинг допустим: {clip_check['is_acceptable']}",
        "",
        "--- СОХРАНЁННЫЙ ФАЙЛ ---",
        f"Путь: {image_path}",
        "",
        "=" * 60,
        "ОТЧЁТ СГЕНЕРИРОВАН АВТОМАТИЧЕСКИ",
        "=" * 60,
    ]

    return "\n".join(report_lines)
