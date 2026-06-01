from __future__ import annotations

from pathlib import Path

from workspace.models import PipelineArtifacts
from workspace.optics.optics_transformer import (
    convert_scene_to_exposure,
    build_default_optics_config
)
from workspace.scene_source.scene_models import (
    get_scene_source_input,
    build_scene_source
)
from workspace.sensor_adc.stubs import (
    build_default_adc_config,
    build_default_charge,
    build_default_frame,
    build_default_sensor_config,
)
from workspace.visualization.visualization import (
    build_default_export_config,
    build_default_preview,
    build_default_reconstruction_config,
    build_default_report,
    export_default_preview,
)
from workspace.shared import (
    demosaic_superpixel,
    normalize_rgb_to_u8,
    save_rgb_bmp,
    summarize_matrix,
)


def main() -> None:
    """
    Точка сборки архитектора.

    Здесь видно, как изолированные роли стыкуются через общие модели:
    scene_source -> optics -> sensor_adc -> visualization.
    """

    scene_input = get_scene_source_input()
    scene_source = build_scene_source(scene_input)

    axis = scene_source.axis
    source = scene_source.source
    object_config = scene_source.object_config
    scene = scene_source.scene

    optics_config = build_default_optics_config()
    #channels = build_default_channels(scene, optics_config)
    #exposure = convert_scene_to_sensor(scene, optics_config, axis)
    exposure = convert_scene_to_exposure(scene, optics_config)

    # sensor_adc: (H,W,3) -> CFA (raw mosaic) -> заряд -> АЦП -> цифровой кадр.
    sensor_config = build_default_sensor_config(exposure)
    charge = build_default_charge(exposure, sensor_config)
    adc_config = build_default_adc_config()
    frame = build_default_frame(charge, adc_config)

    # visualization: демозаик raw-кадра обратно в RGB и экспорт картинки.
    export_config = build_default_export_config()
    rgb = demosaic_superpixel(frame.data)
    image = normalize_rgb_to_u8(rgb)

    output_dir = Path(export_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"final_image.{export_config.image_format}"
    save_rgb_bmp(image, image_path)

    print(f"Raw mosaic (frame): {summarize_matrix(frame.data)}")
    print(f"Saved image: {image_path}")


if __name__ == "__main__":
    main()
