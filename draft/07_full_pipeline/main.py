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
    normalize_frame_to_u8,
    quantize_adc,
    save_grayscale_bmp,
    simulate_scene,
    simulate_sensor,
    spectral_energy_map,
    summarize_matrix,
)


def main() -> None:
    axis = create_uniform_axis(start_nm=400, stop_nm=700, step_nm=20)
    source = gaussian_spectrum(axis, center_nm=560.0, width_nm=90.0, amplitude=1.0)
    reflectance = build_test_reflectance_map(height=32, width=40, axis=axis)

    scene = simulate_scene(axis, source, reflectance)
    scene_energy = spectral_energy_map(scene)

    optical = apply_optical_encoder(scene, axis)
    exposure = optical["sensor_exposure"]
    charge = simulate_sensor(exposure, gain=2000.0)
    frame = quantize_adc(charge, bit_depth=10, full_scale=900.0)
    image = normalize_frame_to_u8(frame)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    bmp_path = output_dir / "final_frame.bmp"
    save_grayscale_bmp(image, bmp_path)

    text = "\n".join(
        [
            "Final pipeline run",
            f"Spectral bands: {axis.band_count}",
            f"Scene energy: {summarize_matrix(scene_energy)}",
            f"Sensor exposure: {summarize_matrix(exposure)}",
            f"Charge: {summarize_matrix(charge)}",
            f"Digital frame: {summarize_matrix(frame)}",
            f"Image: {summarize_matrix(image)}",
            f"Saved image: {bmp_path}",
        ]
    )

    report_path = output_dir / "report.txt"
    dump_text(report_path, text)
    print(text)
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
