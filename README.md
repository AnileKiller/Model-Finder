# Beauty AI — Android Prototype

A Kotlin + Jetpack Compose Android app for on-device AI facial beauty enhancement.
All inference runs locally using TensorFlow Lite — no cloud services required.

## Features

- Select a photo from the gallery
- Face detection with BlazeFace
- 468-point facial landmark extraction with FaceMesh Lite
- Per-pixel skin segmentation with XenoFormer face segmentation
- Real-time beauty sliders (9 controls)
- Side-by-side Original ↔ Enhanced comparison
- Save enhanced image at full resolution

## Quick Start

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd BeautyAI
```

### 2. Generate the Gradle wrapper jar

The `gradle-wrapper.jar` binary is not committed to the repository.
Generate it once with any locally installed Gradle (8.x recommended):

```bash
gradle wrapper --gradle-version 8.7
```

This creates `gradle/wrapper/gradle-wrapper.jar`.

### 3. Place the TFLite models

See **[MODELS.md](MODELS.md)** for model sources and exact filenames.
Copy the three `.tflite` files to:

```
app/src/main/assets/models/
├── blaze_face_short_range.tflite
├── facemesh_lite_468_2022_09_06.f16.tflite
└── face_segmentation_xenoformer_xs_2024_04_02.int8.tflite
```

### 4. Build and install

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Or open the project in Android Studio and press ▶.

## GitHub Actions CI

The workflow in `.github/workflows/build.yml` automatically:
1. Restores TFLite models from GitHub Secrets (base64-encoded)
2. Builds a debug APK
3. Optionally builds and signs a release APK

See **[MODELS.md](MODELS.md)** for instructions on adding model secrets.

## Project Structure

```
app/src/main/java/com/beautyai/prototype/
├── data/
│   ├── inference/          # TFLite wrappers (BlazeFace, FaceMesh, Segmentation)
│   └── repository/         # FaceAnalysisRepository, ImageRepository
├── domain/
│   ├── model/              # BeautyParameters, FaceData, ProcessingState
│   └── usecase/            # AnalyseFaceUseCase, ApplyBeautyUseCase
└── presentation/
    ├── ui/                 # MainActivity, MainScreen, BeautySliders
    │   ├── components/
    │   └── theme/
    └── viewmodel/          # BeautyViewModel
```

## Architecture

- **MVVM** — ViewModel exposes `StateFlow`s; UI is stateless composables
- **Clean layers** — data / domain / presentation are fully decoupled
- **Modular pipeline** — each beauty effect is an isolated function; copy `ApplyBeautyUseCase.kt` and the three inference wrappers into another project with no other dependencies
- **Hardware acceleration** — GPU delegate (when supported), NNAPI, CPU fallback

## Requirements

- Android 8.0 (API 26) or higher
- ~50 MB disk space for models
- Camera/gallery access permission
