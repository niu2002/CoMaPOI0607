from __future__ import annotations

import importlib
from importlib import metadata


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "modelscope",
    "agentscope",
]

OPTIONAL_MODULES = [
    "vllm",
    "faiss",
]


def check_modules(modules, required):
    missing = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"[ok] {module_name}")
        except Exception as exc:
            level = "missing" if required else "optional-missing"
            print(f"[{level}] {module_name}: {exc}")
            missing.append(module_name)
    return missing


def main():
    print("[check] required modules")
    missing_required = check_modules(REQUIRED_MODULES, required=True)
    print("[check] optional modules")
    missing_optional = check_modules(OPTIONAL_MODULES, required=False)

    if "agentscope" not in missing_required:
        try:
            agentscope_version = metadata.version("agentscope")
            print(f"[check] agentscope_version={agentscope_version}")
            major = int(agentscope_version.split(".", 1)[0])
            if major >= 1:
                print("[incompatible] agentscope>=1.0.0 is not supported by this repository")
                print("[incompatible] please install: agentscope>=0.1.0,<1.0.0")
                raise SystemExit(1)
        except metadata.PackageNotFoundError:
            pass

    if missing_required:
        print("[check] result=fail")
        print(f"[check] missing_required={','.join(missing_required)}")
        raise SystemExit(1)

    print("[check] result=pass")
    if missing_optional:
        print(f"[check] missing_optional={','.join(missing_optional)}")


if __name__ == "__main__":
    main()
