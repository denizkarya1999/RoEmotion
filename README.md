# RoEmotion

RoEmotion is a privacy‐preserving, real‐time emotion‐recognition system that uses LED wristbands and a rolling‐shutter smartphone camera to detect and classify four core emotions—Anxiety, Excitement, Sadness, and Anger—without capturing identifiable facial or background information.

## Key Components

- **LED Wristband**  
  Low‐power wristbands emit On–Off Keying (OOK) signals at 6000 Hz, uniquely identifying each wearer and creating fine “streak” patterns when captured by a rolling‐shutter camera.

- **Xemotion App**  
  An Android application that:
  - Controls the phone’s rolling‐shutter rate (up to 6000 Hz) for Computer Vision mode  
  - Switches to AR mode at 250 Hz for headset overlays (e.g., Google Cardboard)  
  - Runs a YOLO‐based object detector to locate wristbands in each frame  
  - Extracts smoothed wrist‐trace lines every 0.7 s  
  - Classifies emotions with a ResNet‐50 neural network trained using a class‐balanced focal loss and OneCycleLR schedule  

- **Modes of Operation**  
  - **Computer Vision Mode:** User‐selectable shutter rate (5 Hz–6000 Hz); on‐screen emotion labels and controls  
  - **Augmented Reality Mode:** Fixed 250 Hz shutter; floating emotion annotations overlaid on the real world  

## Highlights

- **Privacy First:** Only inertial and light‐based wrist data are collected—no video or audio of faces or surroundings.  
- **High Accuracy:** Achieves up to 98.7 % accuracy on negative vs. positive classification and 95 % on fine‐grained emotions.  
- **Low Latency:** Complete end‐to‐end inference (trace extraction, classification, overlay) runs in under 50 ms.  
- **Cost‐Effective AR:** Supports inexpensive smartphone‐based headsets rather than proprietary AR devices.

## Installation & Usage

1. Clone this repository.  
2. Build and install the **Xemotion** Android app on a rolling‐shutter–capable smartphone.  
3. Pair each student’s wristband and select **Computer Vision** or **AR** mode.  
4. Start a session; the app will display emotion labels or AR annotations in real time.  

## Experiment Results

- **Video Stimuli:** Music videos (Guns N’ Roses, Alicia Keys, Destiny’s Child), horror clips (The Backrooms), action scenes, and more.  
- **Lighting Conditions:** Artificial, natural + artificial, and dark rooms.  
- **Metrics:** Binary and four‐way confusion matrices, F1‐score curves, t‐SNE feature clusters.  

## Future Work

- Retrain YOLO detector on 6000 Hz wristband signals for improved robustness.  
- Expand ResNet‐50 training set with more “Sadness” and “Anger” examples.  
- Add battery‐powered on/off switch and direct battery‐to‐Arduino charging.  

---

© 2025 Trustworthy-AI Lab at The University of Michigan-Dearborn
