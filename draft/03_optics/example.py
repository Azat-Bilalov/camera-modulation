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
    summarize_matrix,
)


def main() -> None:
    axis = create_uniform_axis()
    source = gaussian_spectrum(axis, center_nm=560.0, width_nm=90.0, amplitude=1.0)
    reflectance = build_test_reflectance_map(height=20, width=28, axis=axis)
    scene = simulate_scene(axis, source, reflectance)
    optical = apply_optical_encoder(scene, axis)

    text = "\n".join(
        [
            "Stage 3: optical encoder",
            f"Low channel: {summarize_matrix(optical['channel_low'])}",
            f"High channel: {summarize_matrix(optical['channel_high'])}",
            f"Combined exposure: {summarize_matrix(optical['sensor_exposure'])}",
        ]
    )

    output_path = Path(__file__).with_name("output.txt")
    dump_text(output_path, text)
    print(text)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
