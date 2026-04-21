from __future__ import annotations

from pathlib import Path

from workspace.models import PipelineArtifacts
from workspace.optics.stubs import (
    build_default_channels,
    build_default_exposure,
    build_default_optics_config,
)
from workspace.scene_source.stubs import (
    build_default_axis,
    build_default_object,
    build_default_scene,
    build_default_source,
)
from workspace.sensor_adc.stubs import (
    build_default_adc_config,
    build_default_charge,
    build_default_frame,
    build_default_sensor_config,
)
from workspace.visualization.stubs import (
    build_default_export_config,
    build_default_preview,
    build_default_reconstruction_config,
    build_default_report,
    export_default_preview,
)


def main() -> None:
    """
    Точка сборки архитектора.

    Здесь видно, как изолированные роли стыкуются через общие модели:
    scene_source -> optics -> sensor_adc -> visualization.
    """

    axis = build_default_axis()
    source = build_default_source(axis)
    object_config = build_default_object(axis)
    scene = build_default_scene(axis, source, object_config)

    optics_config = build_default_optics_config()
    channels = build_default_channels(scene, optics_config)
    exposure = build_default_exposure(scene, optics_config)

    sensor_config = build_default_sensor_config(exposure)
    charge = build_default_charge(exposure, sensor_config)
    adc_config = build_default_adc_config()
    frame = build_default_frame(charge, adc_config)

    reconstruction_config = build_default_reconstruction_config()
    export_config = build_default_export_config()
    image = build_default_preview(frame, reconstruction_config)
    image_path = export_default_preview(image, export_config)

    artifacts = PipelineArtifacts(
        axis=axis,
        source=source,
        scene=scene,
        exposure=exposure,
        charge=charge,
        frame=frame,
        export=export_config,
        optical_channels=channels,
        description="Интеграционный комплект из role-stub модулей workspace",
    )

    report = build_default_report(artifacts, image_path)
    report_path = Path(export_config.output_dir) / export_config.report_name
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
