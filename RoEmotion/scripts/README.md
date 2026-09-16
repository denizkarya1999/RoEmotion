# September 16, 2026 emotion model exports

The twelve `current_*.pt` assets are the trained weights supplied in the
[RoEmotion Drive folder](https://drive.google.com/drive/folders/1RDEd4dLnu6_7Sp4mP88Co5oQxek6BPxR),
converted from safe state dictionaries into TorchScript for Android's PyTorch
1.13.1 runtime. `model_sources.json` records every source run, exact Drive file
ID, URL, and byte count. The bundled `emotion_models_manifest.json` additionally
records SHA-256 hashes, model contracts, export parity, and runtime verification.

There are three architectures for each of two training splits and two trace
variants. The training splits are Random Distribution and Different Person
Validation; the trace variants are Raw and Kalman Filter + CV. The attention
architecture is the source notebook's `lstm_spatial` model. All source weights
and buffers remain bit-for-bit identical in the exported assets; there is no
quantization or retraining.

## Input and output contract

- Standalone ResNet-50: one float32 tensor shaped `[1, 3, 224, 224]`.
- Recurrent models: one float32 tensor shaped `[1, T, 3, 224, 224]`, with 1–5
  chronological real frames. The app collects five frames for classification.
- Reset recurrent history after a gap exceeding 10 seconds, a participant/video
  boundary, or a model change. Do not mix frames from different trace variants.
- Fit the entire RGB trace image into 224 × 224 with bilinear resizing and
  centered black letterboxing. Divide RGB bytes by 255; normalize channels with
  means `[0.485, 0.456, 0.406]` and deviations `[0.229, 0.224, 0.225]`.
- Outputs are nine logits. Apply softmax once, in class order: Amusement, Anger,
  Anxiety, Disgust, Fear, Joy, Neutral, Sadness, Surprise.

The Kalman Filter + CV choice selects weights trained on that trace variant;
the filter belongs to trace generation, not to the model's neural network.

## Reproduce the export

Use Python with PyTorch 2.12.0, torchvision 0.27.0, and Pillow. The export runs
on CPU. Download each run's `model.pt` and `model_definition.py` from the URLs in
`model_sources.json` into `output/model-release-20260916/source/<id>/`. The
converter checks the original weight file's byte count and reviewed source
definition's SHA-256 before loading anything. Changed Python definitions require
review and an explicit allowlist update. Checkpoints load with `weights_only=True`.

```sh
python3 scripts/convert_emotion_models.py \
  --source-dir output/model-release-20260916/source
```

This writes all twelve assets, the provenance manifest, and the Android test's
`emotion_model_smoke_predictions.json` fixture. It checks exact preservation of
all parameters and buffers, and prediction parity against the supplied Python
model on black frames, distinct random frames, distinct synthetic trace frames,
and (for recurrent models) a one-frame sequence. The dense recurrent wrapper is
equivalent to the original packed LSTM because the app supplies one unpadded
sequence containing only real frames.

## Verify with the app's PyTorch version

Python 3.14 cannot import PyTorch 1.13.1, so the verifier uses that release's
official Linux CPU wheel as a source of LibTorch C++ headers and libraries.
Download and extract the wheel into a temporary directory:

```sh
python3 -m pip download 'torch==1.13.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu \
  --python-version 310 --platform linux_x86_64 --abi cp310 \
  --implementation cp --only-binary=:all: --no-deps \
  --dest /tmp/roemotion-torch-compat
python3 -m zipfile -e '/tmp/roemotion-torch-compat/torch-1.13.1+cpu-cp310-cp310-linux_x86_64.whl' \
  /tmp/roemotion-torch-compat/runtime
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I/tmp/roemotion-torch-compat/runtime/torch/include \
  -I/tmp/roemotion-torch-compat/runtime/torch/include/torch/csrc/api/include \
  scripts/verify_emotion_models.cpp \
  -L/tmp/roemotion-torch-compat/runtime/torch/lib \
  -Wl,-rpath,/tmp/roemotion-torch-compat/runtime/torch/lib \
  -ltorch -ltorch_cpu -lc10 \
  -o /tmp/roemotion-torch-compat/verify_emotion_models
python3 scripts/check_emotion_models_runtime.py \
  --verifier /tmp/roemotion-torch-compat/verify_emotion_models \
  --report output/model-release-20260916/libtorch-1.13.1-parity.json
```

The runtime verifier loads each final asset in LibTorch 1.13.1 and compares its
probabilities with the export runtime on black and distinct random frames, plus
one-frame recurrent sequences. It records results in the report and provenance
manifest. Android instrumented tests separately exercise the same black-frame
fixtures through the app's Java/PyTorch inference path.
