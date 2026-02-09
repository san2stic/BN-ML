from __future__ import annotations

import json

from bn_ml.hardware import hardware_summary, resolve_xgb_device


def main() -> None:
    summary = hardware_summary()
    device_auto, reason_auto = resolve_xgb_device("auto")

    print("Hardware summary")
    print(json.dumps(summary, indent=2))
    print(f"xgb_auto_device: {device_auto}")
    print(f"xgb_auto_reason: {reason_auto}")

    print("\nRecommended config snippet:")
    if device_auto == "cuda":
        print(
            """
model:
  acceleration:
    mode: auto
    cpu_n_jobs: -1
    allow_cuda_fallback: true
""".strip()
        )
    else:
        print(
            """
model:
  acceleration:
    mode: cpu
    cpu_n_jobs: -1
    allow_cuda_fallback: true
""".strip()
        )


if __name__ == "__main__":
    main()
