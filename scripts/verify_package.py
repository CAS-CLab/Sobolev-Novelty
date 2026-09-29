"""Check source syntax, method configuration and portable paths."""

import ast
import json
import subprocess
import sys
from common import ROOT


def main():
    paths = []
    for folder in ["src", "scripts", "configs", "tests", "requirements", "licenses"]:
        paths.extend(
            f for f in (ROOT / folder).rglob("*")
            if f.is_file() and "__pycache__" not in f.parts
        )
    for path in paths:
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == ".json":
            json.loads(path.read_text())
        if path.suffix in [".py", ".json", ".txt", ".md", ".yaml", ".yml", ".sh"]:
            text = path.read_text()
            forbidden_roots = ["/" + name + "/" for name in ("data1", "home")]
            assert not any(prefix in text for prefix in forbidden_roots), path
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/verify_data.py")], check=True
    )
    print(f"OK: {len(paths)} source/configuration files; syntax and portable paths checked.")


if __name__ == "__main__":
    main()
