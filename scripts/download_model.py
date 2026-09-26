"""Optional explicit, resumable download from the official model repository."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_download import ensure_model

parser = argparse.ArgumentParser()
parser.add_argument("--revision", default="790c92633540aa0cb11d9abf19eb46d861714758")
parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "Qwen-Image-2.1")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()
print(f"Downloading official Qwen-Image-2.1 revision {args.revision}", flush=True)
ensure_model(args.directory, revision=args.revision, workers=args.workers)
print(f"Model available at {args.directory.resolve()}", flush=True)
