#!/usr/bin/env python3
"""Export the September 2026 Drive state dictionaries for Android PyTorch.

Usage:
    python3 scripts/convert_emotion_models.py \
        --sources scripts/model_sources.json \
        --source-dir output/model-release-20260916/source

Download each run's model.pt and model_definition.py into source-dir/<id>/.
The supplied Python definitions must be reviewed before this script is run.
The allowlisted hashes below identify the three reviewed source definitions.
No pickle execution, model quantization, retraining, or network access is used.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
from torch import nn


CLASS_NAMES = ["Amusement", "Anger", "Anxiety", "Disgust", "Fear", "Joy",
               "Neutral", "Sadness", "Surprise"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
REVIEWED_DEFINITIONS = {
    "standalone": "89df8f501ea0e7c83cb133ad9b726071446294e85488f9161b6b548730d205a2",
    "lstm": "60cbf6fff8b57ac2baa000b04bf8fae3a536a41ecdb1c165674ce4533a34982f",
    "lstm_attention": "1d7b62e0eb4fbeb916c1c2c4c76d799732b9d7e6d7acd29d5c533ba68329d50d",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StandaloneInference(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.cnn = original.cnn
        self.pool = original.pool
        self.classifier = original.classifier

    def forward(self, images):
        maps = self.cnn(images.contiguous(memory_format=torch.channels_last))
        return self.classifier(self.pool(maps).flatten(1))


class RecurrentInference(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.cnn = original.cnn
        self.spatial_attention = getattr(original, "spatial_attention", nn.Identity())
        self.pool = original.pool
        self.lstm = original.lstm
        self.classifier = original.classifier

    def forward(self, images):
        # Every supplied frame is real. For the app's unpadded batch of one,
        # dense LSTM and the training code's packed LSTM have identical semantics.
        batch, steps = images.shape[0], images.shape[1]
        frames = images.reshape(-1, 3, 224, 224)
        maps = self.cnn(frames.contiguous(memory_format=torch.channels_last))
        features = self.pool(self.spatial_attention(maps)).flatten(1)
        _, (hidden, _) = self.lstm(features.reshape(batch, steps, 2048))
        return self.classifier(hidden[-1])


def input_cases(steps, recurrent):
    generator = torch.Generator().manual_seed(20260916)
    raw = torch.rand((steps, 3, 224, 224), generator=generator)
    black = torch.zeros_like(raw)
    # Distinct chronological synthetic trace frames, including all channels.
    traces = torch.zeros_like(raw)
    for step in range(steps):
        for x in range(20, 205):
            y = 40 + ((x * (step + 2)) % 135)
            traces[step, :, max(0, y-2):y+3, x] = torch.tensor([1., .75, .25])[:, None]
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    for name, pixels in (("black", black), ("random_distinct_frames", raw),
                         ("synthetic_distinct_traces", traces)):
        normalized = (pixels - mean) / std
        yield name, normalized.unsqueeze(0) if recurrent else normalized


def convert(source, args):
    started = time.monotonic()
    folder = args.source_dir / source["id"]
    checkpoint_path = folder / "model.pt"
    definition_path = folder / "model_definition.py"
    expected_size = int(next(f["size"] for f in source["files"] if f["name"] == "model.pt"))
    if checkpoint_path.stat().st_size != expected_size:
        raise ValueError(f"Incomplete source download: {checkpoint_path}")
    definition_hash = sha256(definition_path)
    if definition_hash != REVIEWED_DEFINITIONS[source["mode"]]:
        raise ValueError(f"Definition changed; review before execution: {definition_path}")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    settings = saved["settings"]
    recurrent = source["mode"] != "standalone"
    steps = 5 if recurrent else 1
    mode = "lstm_spatial" if source["mode"] == "lstm_attention" else source["mode"]
    assert saved["label_mode"] == "single_label"
    assert saved["mode"] == mode
    assert saved["class_names"] == CLASS_NAMES
    assert settings["sequence_length"] == steps
    assert settings["max_gap_seconds"] == 10.0
    assert settings["image_size"] == 224
    assert settings["preprocessing"] == "RGB letterbox black; ImageNet normalization"

    spec = importlib.util.spec_from_file_location("reviewed_" + source["id"], definition_path)
    definition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definition)
    original = definition.EmotionModel(mode, num_classes=9, hidden_size=saved["hidden_size"],
                                       pretrained=False).eval()
    original.load_state_dict(saved["model_state"], strict=True)
    wrapper = (RecurrentInference(original) if recurrent else StandaloneInference(original)).eval()
    cases = list(input_cases(steps, recurrent))
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, cases[0][1], check_trace=False, strict=True)
        asset_path = args.assets / ("current_" + source["id"] + ".pt")
        torch.jit.save(traced, str(asset_path))
        reloaded = torch.jit.load(str(asset_path), map_location="cpu").eval()
        # Serialized parameters/buffers must preserve every source bit.
        exported_state = reloaded.state_dict()
        assert set(exported_state) == set(saved["model_state"])
        assert all(torch.equal(value, exported_state[key])
                   for key, value in saved["model_state"].items())
        parity = []
        probabilities = None
        for name, images in cases:
            expected = original(images)
            actual = reloaded(images)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
            assert torch.isfinite(actual).all()
            error = float((actual - expected).abs().max())
            parity.append({"case": name, "max_abs_logit_error": error,
                           "same_predicted_class": bool(actual.argmax(1).eq(expected.argmax(1)).all())})
            if name == "black":
                probabilities = actual.softmax(1)[0].tolist()
        # Ensure traced tensor shapes preserve the training-supported short history.
        if recurrent:
            short = cases[1][1][:, :1]
            expected = original(short)
            actual = reloaded(short)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
            parity.append({"case": "single_real_frame", "max_abs_logit_error": float((actual-expected).abs().max()),
                           "same_predicted_class": bool(actual.argmax(1).eq(expected.argmax(1)).all())})

    entry = {"id": source["id"], "asset": asset_path.name,
             "training_split": source["split"], "architecture": source["mode"],
             "trace_variant": source["trace"], "source_run_name": source["runName"],
             "source_run_url": "https://drive.google.com/drive/folders/" + source["runId"],
             "source_files": source["files"], "source_sha256": sha256(checkpoint_path),
             "definition_sha256": definition_hash, "asset_sha256": sha256(asset_path),
             "asset_bytes": asset_path.stat().st_size, "class_names": CLASS_NAMES,
             "input_shape": [1, steps, 3, 224, 224] if recurrent else [1, 3, 224, 224],
             "sequence_length": steps, "minimum_sequence_length": 1,
             "max_gap_seconds": settings["max_gap_seconds"],
             "preprocessing": {"size": 224, "color": "RGB", "resize": "bilinear letterbox",
                               "padding": "black", "mean": MEAN, "std": STD},
             "outputs": "nine logits; apply softmax once", "best_epoch": saved["best_epoch"],
             "training_dataset_sha256": settings["archive_sha256"],
             "split_identity": settings.get("split_identity"), "conversion_parity": parity,
             "weights_preserved_bit_exact": True, "operators": sorted(torch.jit.export_opnames(reloaded))}
    print(json.dumps({"id": source["id"], "bytes": entry["asset_bytes"],
                      "seconds": round(time.monotonic()-started, 1), "parity": parity}), flush=True)
    return entry, {"sequence_length": steps, "probabilities": probabilities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("scripts/model_sources.json"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("app/src/main/assets"))
    parser.add_argument("--smoke-fixtures", type=Path,
                        default=Path("app/src/androidTest/assets/emotion_model_smoke_predictions.json"))
    parser.add_argument("--only", help="Comma-separated model IDs for partial export")
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Both the original packed path and exported dense path then use ordinary
    # float32 ATen operators, as available in Android's PyTorch 1.13.1 runtime.
    torch.backends.mkldnn.enabled = False
    args.assets.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    if args.only:
        selected = set(args.only.split(","))
        sources = [source for source in sources if source["id"] in selected]
    entries, fixtures = [], {}
    for source in sources:
        entry, fixture = convert(source, args)
        entries.append(entry)
        fixtures["current_" + source["id"]] = fixture["probabilities"]
    manifest = {"format_version": 1, "release_date": "2026-09-16",
                "source_folder": "https://drive.google.com/drive/folders/1RDEd4dLnu6_7Sp4mP88Co5oQxek6BPxR",
                "export_torch_version": torch.__version__, "target_android_pytorch_version": "1.13.1",
                "precision": "float32, no quantization", "models": entries}
    (args.assets / "emotion_models_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.smoke_fixtures.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_fixtures.write_text(json.dumps(fixtures, indent=2) + "\n")


if __name__ == "__main__":
    main()
