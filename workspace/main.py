from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from workspace.models import (
    ChargeMatrix,
    DigitalFrame,
    PipelineArtifacts,
)
from workspace.optics.optics_transformer import (
    build_default_optics_config,
    convert_scene_to_exposure,
)
from workspace.scene_source.scene_models import (
    build_scene_source,
    get_scene_source_input,
)
from workspace.sensor_adc.sensor_pipeline import (
    build_default_adc_config,
    build_default_sensor_config,
    build_rgb_frame,
)
from workspace.shared import (
    normalize_rgb_to_u8,
    save_rgb_bmp,
    summarize_matrix,
)
from workspace.visualization.visualization import (
    build_default_export_config,
    build_default_report,
)


def _save_upscaled_png(rgb_image: list[list[list[int]]], path: Path, scale: int = 8) -> None:
    """Сохраняет RGB-изображение в PNG с nearest-neighbor upscaling."""
    height = len(rgb_image)
    width = len(rgb_image[0])
    # PIL принимает плоский список пикселей в формате (R, G, B)
    flat_pixels = [(pixel[0], pixel[1], pixel[2]) for row in rgb_image for pixel in row]
    img = Image.new("RGB", (width, height))
    img.putdata(flat_pixels)
    img = img.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    img.save(path)


def main() -> None:
    """
    Точка сборки архитектора.

    Пример запуска:
        python -m workspace.main
        python -m workspace.main --spectrum workspace/input/scarlet_spectrum.csv
    """
    parser = argparse.ArgumentParser(description="Спектральный симулятор формирования изображения.")
    parser.add_argument(
        "--spectrum",
        type=Path,
        default=Path(__file__).resolve().parent / "input" / "sample_spectrum.csv",
        help="Путь к CSV-файлу со спектром (колонка 'value').",
    )
    args = parser.parse_args()

    scene_input = get_scene_source_input(str(args.spectrum))
    scene_source = build_scene_source(scene_input)

    axis = scene_source.axis
    source = scene_source.source
    scene = scene_source.scene

    optics_config = build_default_optics_config()
    exposure = convert_scene_to_exposure(scene, optics_config)

    # sensor_adc: (H,W,3) -> заряд -> АЦП для каждого канала отдельно (без Bayer).
    sensor_config = build_default_sensor_config(exposure)
    adc_config = build_default_adc_config()
    rgb_int = build_rgb_frame(exposure, sensor_config, adc_config)

    # Для отчёта формируем усреднённый grayscale-кадр из RGB.
    grayscale_data = [[sum(pixel) // 3 for pixel in row] for row in rgb_int]
    frame = DigitalFrame(
        data=grayscale_data,
        bit_depth=adc_config.bit_depth,
        description="Усреднённый grayscale кадр для отчёта",
    )

    # visualization: нормализация RGB и экспорт.
    export_config = build_default_export_config()
    image = normalize_rgb_to_u8(rgb_int)

    output_dir = Path(export_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"final_image.{export_config.image_format}"
    save_rgb_bmp(image, image_path)

    # Upscaled PNG для наглядности
    png_path = output_dir / "final_image_256x256.png"
    _save_upscaled_png(image, png_path, scale=8)

    # Сборка артефактов для отчёта.
    artifacts = PipelineArtifacts(
        axis=axis,
        source=source,
        scene=scene,
        exposure=exposure,
        charge=ChargeMatrix(
            charge=[[float(v) for v in row] for row in grayscale_data],
            sensor_config=sensor_config,
        ),
        frame=frame,
        export=export_config,
    )
    report = build_default_report(artifacts, image_path)
    report_path = output_dir / export_config.report_name
    report_path.write_text(report, encoding="utf-8")

    print(f"RGB frame (raw): size={len(rgb_int)}x{len(rgb_int[0])}, channels=3")
    print(f"Grayscale (report): {summarize_matrix(frame.data)}")
    print(f"Saved BMP: {image_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
