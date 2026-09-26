"""Build the portable bootstrap package; never include weights or user data."""
from pathlib import Path
import hashlib
import zipfile

root = Path(__file__).resolve().parent
version = (root / "VERSION").read_text(encoding="ascii").strip()
files = ["Start.cmd", "start.ps1", "service.py", "model_download.py", "index.html", "requirements.txt",
         "requirements-quant.txt", "settings.example.json", "README.md", "LICENSE", "NOTICE.md", "VERSION",
         "docs/validation.md", "scripts/download_model.py", "scripts/qa_gpu.py", "examples/client.py",
         "docs/examples/teapot.png", "docs/examples/edit.png", "docs/examples/star.png"]
destination = root / "dist" / f"Dreamer-Local-Image-NVIDIA-{version}.zip"
destination.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        archive.write(root / name, "Dreamer-Local-Image/" + name)
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
destination.with_suffix(".zip.sha256").write_text(digest + "  " + destination.name + "\n", encoding="ascii")
print(destination)
print("SHA256 " + digest)
