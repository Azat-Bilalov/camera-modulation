from __future__ import annotations

from pathlib import Path

from workspace.models import PipelineArtifacts
from workspace.optics.stubs import (
    build_default_optics_config,
)
from workspace.optics.optics_transformer import convert_scene_to_sensor

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

from workspace.scene_source import (
    build_axis,
    build_optic_input,
    parse_list,
    SpectralAxis,
    SourceConfig,
    ObjectConfig
)

from workspace.optics import (
    convert_scene_to_sensor,
    convert_scene_to_channels
)

def build_axis_by_step(start: float, step: float, count: int) -> SpectralAxis:
    wave = [start + i * step for i in range(count)]
    return SpectralAxis(
        wave=wave,
        start=wave[0],
        stop=wave[-1],
        bands_count=len(wave)
    )


def main() -> None:
    """
    Точка сборки архитектора.

    Здесь видно, как изолированные роли стыкуются через общие модели:
    scene_source -> optics -> sensor_adc -> visualization.
    """

    #wave = parse_list(input("Длины волн, нм: "))
    #radiation = parse_list(input("Мощность излучения: "))
    #coef = parse_list(input("Коэффициенты отражения объекта: "))

    axis = build_axis_by_step(
        start=380.0,
        step=10.0,
        count=36
    )
    #source_config = SourceConfig(spectrum=radiation)
    #object_config = ObjectConfig(reflectance=coef)
    optic_input = list[7.19772, 8.66835, 10.0241, 10.6711, 10.9908, 11.318, 11.713, 12.13, 12.5267, 12.8175, 13.015, 13.1402, 13.2771, 13.4178, 13.5352, 13.54, 13.4372, 13.1751, 12.6869, 12.1096, 11.3222, 10.3386, 9.22227, 8.32678, 7.86045, 7.65831, 7.57516, 7.56342, 7.72733, 8.0386, 8.40108, 8.71172, 8.92745, 9.01649, 9.01843, 9.165]

    optics_config = build_default_optics_config()
    exposure = convert_scene_to_sensor(optic_input, optics_config, axis)

    sensor_config = build_default_sensor_config(exposure)
    charge = build_default_charge(exposure, sensor_config)
    adc_config = build_default_adc_config()
    frame = build_default_frame(charge, adc_config)

    reconstruction_config = build_default_reconstruction_config()
    export_config = build_default_export_config()
    image = build_default_preview(frame, reconstruction_config)
    image_path = export_default_preview(image, export_config)

    # artifacts = PipelineArtifacts(
    #     axis=axis,
    #     source=source,
    #     scene=scene,
    #     exposure=exposure,
    #     charge=charge,
    #     frame=frame,
    #     export=export_config,
    #     optical_channels=channels,
    #     description="Интеграционный комплект из role-stub модулей workspace",
    # )

    # report = build_default_report(artifacts, image_path)
    # report_path = Path(export_config.output_dir) / export_config.report_name
    # report_path.write_text(report, encoding="utf-8")

    #print(report)
    #print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
