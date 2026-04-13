from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import create_uniform_axis, dump_text


def main() -> None:
    axis = create_uniform_axis(start_nm=400, stop_nm=700, step_nm=20)
    text = "\n".join(
        [
            "Stage 1: spectral axis",
            f"Band count: {axis.band_count}",
            f"Wavelengths (nm): {', '.join(str(int(value)) for value in axis.wavelengths_nm)}",
        ]
    )
    output_path = Path(__file__).with_name("output.txt")
    dump_text(output_path, text)
    print(text)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
