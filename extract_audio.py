"""Extract WAV audio from MP4 files with the external FFmpeg CLI."""

import argparse
from pathlib import Path
import subprocess


def extract_audio(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(input_dir.glob("*.mp4")):
        output = output_dir / (source.stem + ".wav")
        subprocess.run(
            ["ffmpeg", "-n", "-i", str(source.resolve()), "-vn", str(output.resolve())],
            check=True,
            timeout=1800,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract_audio(args.input, args.output)
