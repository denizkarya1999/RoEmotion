# Cardboard AR mode

In **Inference Mode**, tap **AR Mode**, choose 1–1440 minutes,
then tap **Start**. Insert the phone horizontally with the middle of the screen between
the viewer lenses and the rear camera unobstructed. Setup values are remembered.
A front-camera session is stopped before switching to the rear camera.

The countdown and processed camera frame are drawn identically for each eye.
The camera image always fills each eye’s entire half of the screen, with a centered
crop instead of black bands. There is no drawn center divider or image-size control;
old saved image-size preferences no longer affect rendering. Its aspect ratio is
preserved, so some side content is cropped on landscape phones.
During normal viewing, the timer and exit hint are hidden; countdown, rotation
and camera-waiting messages appear only when needed. The timer still ends the
session automatically. Recognition runs once per frame; the display shares
that bitmap between the two eyes. The existing 1280-pixel processing limit remains.

## Shared camera settings and frame processing

ISO, shutter frequency (Hz), the custom ISO switch, flash and lighting use the same
saved preferences in AR and CV. AR no longer forces ISO 100 and 6000 Hz. Values
are still clamped to the camera’s supported ranges; zoom is retained when entering
and exiting AR. Detection settings remain shared; only the appearance controls below
are separate.

The processor reuses the mutable preview bitmap for its output in both detection
pipelines, avoiding one full-frame bitmap allocation per frame (about 3.5 MiB at
1280×720). Resolution and model inputs are unchanged. Immutable inputs retain a
separate output, and ownership passes to the display before recycling. Native YOLO
work stays on the existing processing worker without a second dispatcher handoff.

## Separate appearance settings

Open **Settings → CV Mode Appearance** for the handheld display or
**Settings → AR Mode Appearance** for the headset. Each has independent controls
for **Box Line Thickness**, **Text Size**, and **Trace Line Thickness**.

| Setting | CV default | AR default |
| --- | ---: | ---: |
| Box line thickness | 10 | 3 |
| Text size | 16 | 8 |
| Trace line thickness | 5 | 2 |

Line controls range from 1–20; text size ranges from 4–20. Values are saved per
mode. AR starts smaller and slimmer, but each setting can be changed independently.
There is no hidden multiplier between CV and AR values. Both YOLO and contour
rendering receive a snapshot of the chosen profile for each frame.

Existing inference-label and trace-boldness preferences seed CV appearance once;
AR starts with its own defaults. Existing new-profile values are never overwritten.
**Saved Trace Appearance → Saved Trace Thickness** controls collected/model-input
traces separately. Changing the live AR/CV appearance does not change those inputs.
Detector bounds and camera zoom are not changed by appearance controls.

Hold a touch-type Cardboard trigger or long-press the screen to exit. Android Back
also exits, and the session expires automatically. Older magnetic-only triggers
are not supported. Leaving the app cancels headset mode and stops processing.
Returning to the app requires starting a new session. Exiting AR restores the
previous orientation policy and handheld controls. A session
already running when AR was entered continues on timer/manual exit; a session
started by AR is stopped on exit.

Rotation is handled without recreating the camera/inference activity. Results
from before a resize or session stop are discarded. Images older than two seconds
are hidden and replaced with a waiting message. No inference or camera work is
performed per eye.

## Scope and physical verification

This is a **side-by-side binocular monitor of one camera**, not stereoscopic depth,
world-anchored AR, or a full Cardboard SDK integration. The full-screen crop is not optical calibration. Viewer QR profiles, lens distortion correction,
physical lens-spacing calibration and magnetic trigger sensing are not implemented.
Google's [Cardboard SDK](https://developers.google.com/cardboard/develop/c/quickstart)
provides viewer-specific distortion for a future calibrated rendering path.

Before relying on a particular phone/viewer combination, check the following on
the real hardware (automated rendering tests cannot verify optics):

- Camera opening, comfortable image size, text readability and alignment in both eyes.
- Landscape and reverse-landscape preview orientation and wristband label placement.
- Touch-trigger hold, manual Back, expiry, Home/resume, and phone removal.
- Camera-to-display latency and sustained inference performance/temperature.

## Automated checks

`CardboardLayoutTest` verifies equal eye geometry, aspect preservation, full viewport coverage,
symmetric cropping and rejected invalid input. `CardboardInstrumentedTest` verifies
failed startup, Back/long-press exit, ownership of an existing inference session,
background cancellation, expiry, duplicate entry and identical rendered eyes.
The instrumentation host is included only in debug builds and does not load models
or open the camera automatically. `SharedCameraSettingsInstrumentedTest` opens a
camera only to inspect real request settings across AR transitions, without a
capture session. `FrameReuseInstrumentedTest` verifies bitmap reuse, pixel output,
immutable-input fallback and display ownership after asynchronous cleanup.

```sh
bash gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug :app:assembleDebugAndroidTest
bash gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.developer27.xemotion.ar.CardboardInstrumentedTest
```
