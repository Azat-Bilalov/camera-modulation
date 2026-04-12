from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import (
    build_test_reflectance_map,
    create_uniform_axis,
    dump_text,
    gaussian_spectrum,
    simulate_scene,
    spectral_energy_map,
    summarize_matrix,
)


def main() -> None:
    axis = create_uniform_axis()
    source = gaussian_spectrum(axis, center_nm=560.0, width_nm=90.0, amplitude=1.0)
    reflectance = build_test_reflectance_map(height=20, width=28, axis=axis)
    scene = simulate_scene(axis, source, reflectance)
    energy = spectral_energy_map(scene)

    center_pixel = scene[len(scene) // 2][len(scene[0]) // 2]
    text = "\n".join(
        [
            "Stage 2: scene spectral image",
            f"Scene summary: {summarize_matrix(energy)}",
            f"Center pixel spectral bands: {[round(value, 4) for value in center_pixel]}",
        ]
    )

    output_path = Path(__file__).with_name("output.txt")
    dump_text(output_path, text)
    print(text)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
