#!/usr/bin/env python3
"""Compare all exports with a LibTorch 1.13.1 build of verify_emotion_models.cpp.

Pass --verifier /path/to/verify_emotion_models after building against the official
PyTorch 1.13.1+cpu wheel's C++ headers and libraries. This verifies old-runtime
loading and inference on both black frames and distinct normalized frame inputs.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from convert_emotion_models import input_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("app/src/main/assets/emotion_models_manifest.json"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = False
    manifest = json.loads(args.manifest.read_text())
    results = []
    with tempfile.TemporaryDirectory(prefix="roemotion-compat-") as directory, torch.inference_mode():
        input_file = Path(directory) / "input.bin"
        for entry in manifest["models"]:
            asset = args.manifest.parent / entry["asset"]
            assert hashlib.sha256(asset.read_bytes()).hexdigest() == entry["asset_sha256"]
            current = torch.jit.load(str(asset), map_location="cpu").eval()
            steps = entry["sequence_length"]
            recurrent = entry["architecture"] != "standalone"
            cases = list(input_cases(steps, recurrent))[:2]
            if recurrent:
                cases.append(("single_real_frame", cases[1][1][:, :1]))
            checks = []
            for case_name, images in cases:
                images.contiguous().numpy().tofile(input_file)
                case_steps = int(images.shape[1]) if recurrent else 1
                command = [str(args.verifier), str(asset), str(case_steps), str(input_file)]
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                actual = torch.tensor(json.loads(process.stdout))
                expected = current(images).softmax(1)[0]
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
                check = {"case": case_name,
                         "max_abs_probability_error": float((actual-expected).abs().max()),
                         "same_predicted_class": bool(actual.argmax().eq(expected.argmax()))}
                assert check["same_predicted_class"]
                checks.append(check)
            result = {"id": entry["id"], "asset_sha256": entry["asset_sha256"], "checks": checks}
            results.append(result)
            entry["libtorch_1_13_1_verification"] = checks
            print(json.dumps(result), flush=True)
    report = {"runtime": "official PyTorch 1.13.1+cpu Linux LibTorch",
              "tolerance": {"relative": 1e-4, "absolute": 2e-5}, "models": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    manifest["verified_compatibility_runtime"] = report["runtime"]
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
