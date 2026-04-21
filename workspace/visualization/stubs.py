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
    Собирает краткий текстовый отчет для архитектора и роли верификации.

    Input:
    - `artifacts`: собранный комплект сущностей pipeline.
    - `image_path`: путь к сохраненному preview-файлу.

    Output:
    - строка отчета, которую можно сохранить рядом с картинкой.

    Где смотреть пример:
    - `draft/07_full_pipeline/outputs/report.txt`
    """

    return "\n".join(
        [
            "Workspace integration run",
            f"Spectral bands: {artifacts.axis.band_count}",
            f"Optical channels: {[channel.channel_id for channel in artifacts.optical_channels]}",
            f"Sensor exposure: {summarize_matrix(artifacts.exposure.irradiance)}",
            f"Charge: {summarize_matrix(artifacts.charge.charge)}",
            f"Digital frame: {summarize_matrix(artifacts.frame.data)}",
            f"Saved image: {image_path}",
        ]
    )
