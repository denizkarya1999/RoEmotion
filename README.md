# RoEmotion

RoEmotion is a privacy-preserving Android emotion-recognition system that uses
LED wristbands and a rolling-shutter smartphone camera. It detects the light
traces produced by each wristband and runs emotion classification locally on
the device, without relying on identifiable facial imagery or cloud inference.

## RoEmotion 1.0

Version 1.0 was released on **September 16th 2026**. The app includes thirteen
offline emotion models: the previous ResNet-50 + LSTM + Attention model and
twelve current models trained across two validation strategies, three neural
network architectures, and two trace-processing variants.

The first item on the **Options** screen is **Change Emotion Detection Model**.
The selection hierarchy is:

```text
Change Emotion Detection Model
├── Previous Models
│   └── ResNet-50 + LSTM + Attention Model
└── Current Models
    ├── Trained with Random Distribution
    │   ├── Standalone ResNet-50
    │   ├── ResNet-50 + LSTM
    │   └── ResNet-50 + LSTM + Attention
    └── Trained with Different Person Validation
        ├── Standalone ResNet-50
        ├── ResNet-50 + LSTM
        └── ResNet-50 + LSTM + Attention
```

After choosing a current model, select either **Raw** or **Kalman Filter + CV**.
The app remembers the completed selection, automatically uses the matching
trace processing, and reloads inference without requiring a restart.

## Emotion Classification

The current models classify nine emotions in this order:

- Amusement
- Anger
- Anxiety
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Surprise

Standalone models classify a single frame. The LSTM models accept sequences of
one to five real frames, with five frames collected during live inference. All
current models use 224 × 224 RGB input, black letterboxing, and ImageNet
normalization. The bundled model assets run through PyTorch 1.13.1 for Android.

## App Features

- Fully offline model selection and inference
- Rolling-shutter LED wristband detection and trace extraction
- Computer Vision and Augmented Reality display modes
- Model switching through the Options screen
- Persistent model selection across app launches
- Light and dark themes, including light-blue text on dark Options dialogs
- Local per-user trace filtering for Kalman Filter + CV models

## Installation

The release APK is named **Install RoEmotion 1.0.apk**. It is approximately
1.70 GB because all thirteen model variants are bundled for offline use.

To build the app from source:

```sh
cd RoEmotion
./gradlew assembleRelease
```

The project targets Android SDK 36 and supports Android 8.0 (API 26) or newer.
The release APK retains the project's existing signing configuration so it can
update compatible local installations.

## Verification

RoEmotion 1.0 passed:

- 42 JVM unit tests
- Android lint with no errors
- Debug, device-test, and release builds
- 11 instrumentation tests on a physical Google Pixel 8a running Android 17
- All thirteen model input/output contract checks
- Exact source-to-export prediction parity across 44 test cases
- PyTorch 1.13.1 runtime compatibility checks across 32 test cases
- SHA-256 verification for every packaged current-model asset
- APK signature verification

See the [release notes](RoEmotion/docs/releases/2026-09-16.md) for the complete
model behavior and test record. Model provenance, conversion, and reproduction
instructions are documented in [RoEmotion/scripts/README.md](RoEmotion/scripts/README.md).

## Developers

- Deniz Acikbas
- Hasky Fynn
- Thuy-Anh Nguyen
- Amina Bilal
- Christian Nwobu

---

© 2026 Trustworthy AI Lab at the University of Michigan-Dearborn
