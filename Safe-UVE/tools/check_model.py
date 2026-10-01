from pathlib import Path
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Check Safe-UVE YOLO model classes.")
    parser.add_argument("--model", default="models/emergency_vehicle.pt")
    args = parser.parse_args()
    path = Path(args.model)
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    model = YOLO(str(path))
    print("Model:", path)
    print("Classes:", model.names)


if __name__ == "__main__":
    main()
