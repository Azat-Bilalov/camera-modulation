from __future__ import annotations

from pathlib import Path

from workspace.models import ExportConfig, PipelineArtifacts
from workspace.shared import summarize_matrix


def build_default_export_config(output_dir: str | None = None) -> ExportConfig:
    """
    Возвращает настройки сохранения итогового кадра и служебных артефактов.
    """

    target_dir = output_dir or str(Path(__file__).resolve().parents[1] / "outputs")
    return ExportConfig(
        output_dir=target_dir,
        image_format="bmp",
        save_intermediate=True,
        report_name="workspace_report.txt",
        description="Заглушка параметров экспорта для роли visualization",
    )


def build_default_report(artifacts: PipelineArtifacts, image_path: Path) -> str:
    """
    Собирает подробный текстовый отчет для архитектора и роли верификации.
    """
    from workspace.visualization.verifier import (
        calculate_image_statistics,
        verify_digital_range,
        verify_no_clipping,
    )

    frame_stats = calculate_image_statistics(
        artifacts.frame.data, artifacts.frame.bit_depth
    )
    range_check = verify_digital_range(artifacts.frame.data, artifacts.frame.bit_depth)
    clip_check = verify_no_clipping(artifacts.frame.data, artifacts.frame.bit_depth)

    exposure_data = artifacts.exposure.channel_irradiance
    if exposure_data:
        flat = [v for row in exposure_data for pixel in row for v in pixel]
        exposure_summary = (
            f"size={len(exposure_data)}x{len(exposure_data[0])}x3, "
            f"min={min(flat):.4f}, max={max(flat):.4f}, mean={sum(flat) / len(flat):.4f}"
        )
    else:
        exposure_summary = "N/A"

    report_lines = [
        "=" * 60,
        "ОТЧЁТ ПО РАБОТЕ МОДУЛЯ ВИЗУАЛИЗАЦИИ И ЭКСПОРТА",
        "=" * 60,
        "",
        "--- ОСНОВНАЯ ИНФОРМАЦИЯ ---",
        f"Спектральных диапазонов: {artifacts.axis.bands_count}",
        f"Оптические каналы: {[ch.channel_id for ch in artifacts.optical_channels]}",
        "",
        "--- ПАРАМЕТРЫ СЕНСОРА ---",
        f"Экспозиция: {exposure_summary}",
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
