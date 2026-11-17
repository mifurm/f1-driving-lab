"""Package the completed lab without installed environments or browser binaries."""

from pathlib import Path
import zipfile


def bundle():
    root = Path(__file__).resolve().parent
    destination = root / "f1-driving-lab.zip"
    excluded = {".venv", ".browsers", "node_modules", "__pycache__", ".git"}
    if not (root / "artifacts/verification.json").exists():
        raise FileNotFoundError("Run verify.py before bundling the completed lab")
    paths = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path != destination
        and not any(part in excluded for part in path.relative_to(root).parts)
        and path.name != ".DS_Store"
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Portable bundle failed CRC verification")
        required = [
            "README.md", "PLAN.md", "driving.py", "train.py", "render.py",
            "artifacts/replay.html", "artifacts/training_deck.pptx",
            "artifacts/models/final.npz", "artifacts/data/aggregated_training_data.npz",
            "artifacts/verification.json", "artifacts/browser_verification.json",
        ]
        for path in required:
            archive.getinfo(str(Path(root.name) / path))
        for number in range(121):
            archive.getinfo(str(Path(root.name) / f"artifacts/trajectories/round_{number:03d}.npz"))
    print(f"Verified bundle: {destination} ({destination.stat().st_size / 1e6:.1f} MB; {len(paths)} files)")


if __name__ == "__main__":
    bundle()
