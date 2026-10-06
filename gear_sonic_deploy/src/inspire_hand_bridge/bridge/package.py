"""Package the existing native driver for an independent Linux bridge build."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def package(output):
    native = Path(__file__).resolve().parents[1]
    repo = native.parents[2]
    files = {
        "source/CMakeLists.txt": native / "CMakeLists.txt",
        "run.sh": native / "bridge/run.sh",
        "bridge.env.example": native / "bridge/bridge.env.example",
        "LICENSE": repo / "LICENSE",
        "NOTICE.md": repo / "legal/NOTICE-inspire-hands.md",
    }
    for directory in ("include", "src"):
        for path in sorted((native / directory).rglob("*")):
            if path.is_file():
                files["source/" + str(path.relative_to(native))] = path
    for name in ("LICENSE-g1_ros.txt", "LICENSE-unitree_mujoco.txt", "LICENSE-unitree_ros.txt"):
        files[name] = repo / "legal" / name
    manifest = {"format": "inspire-bridge-source-v1", "files": {}}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            manifest["files"][name] = {
                "source": str(path.relative_to(repo)),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name == "run.sh" else 0o644
            archive.addfile(info, io.BytesIO(data))
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("source-manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new .tar.gz path")
    package(parser.parse_args().output)
