from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import (
    apply_optical_encoder,
    build_test_reflectance_map,
    create_uniform_axis,
    dump_text,
    gaussian_spectrum,
    simulate_scene,
    simulate_sensor,
    summarize_matrix,
)


def main() -> None:
    axis = create_uniform_axis()
    source = gaussian_spectrum(axis, center_nm=560.0, width_nm=90.0, amplitude=1.0)
    reflectance = build_test_reflectance_map(height=20, width=28, axis=axis)
    scene = simulate_scene(axis, source, reflectance)
    exposure = apply_optical_encoder(scene, axis)["sensor_exposure"]
    charge = simulate_sensor(exposure, gain=1900.0)

    text = "\n".join(
        [
            "Stage 4: sensor model",
            f"Exposure summary: {summarize_matrix(exposure)}",
            f"Charge summary: {summarize_matrix(charge)}",
        ]
    )

    output_path = Path(__file__).with_name("output.txt")
    dump_text(output_path, text)
    print(text)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
