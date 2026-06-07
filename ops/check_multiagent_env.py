from __future__ import annotations

import importlib


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

    if missing_required:
        print("[check] result=fail")
        print(f"[check] missing_required={','.join(missing_required)}")
        raise SystemExit(1)

    print("[check] result=pass")
    if missing_optional:
        print(f"[check] missing_optional={','.join(missing_optional)}")


if __name__ == "__main__":
    main()
