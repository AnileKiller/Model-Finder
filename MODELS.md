# TFLite Model Files

The three AI models are **not included in this repository** because binary files
should not be committed to Git. Place them in the following folder before
building the app:

```
app/src/main/assets/models/
├── blaze_face_short_range.tflite
├── facemesh_lite_468_2022_09_06.f16.tflite
└── face_segmentation_xenoformer_xs_2024_04_02.int8.tflite
```

## Where to get the models

### BlazeFace (short-range)
- Official MediaPipe repository:  
  `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite`

### FaceMesh Lite (468 landmarks, f16)
- MediaPipe Face Landmark task model:  
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`  
  (The `.tflite` inside the task bundle is named `facemesh_lite_468_2022_09_06.f16.tflite`)
- Or from the MediaPipe Solutions download page:  
  `https://developers.google.com/mediapipe/solutions/vision/face_landmarker`

### Face Segmentation (XenoFormer XS, int8)
- This model may be bundled with the MediaPipe Face Stylizer or a custom
  build. If you obtained it from a third-party source place it here with the
  exact filename shown above.

---

## GitHub Actions (CI/CD)

The included workflow (`.github/workflows/build.yml`) restores models from
**GitHub Secrets** at build time so they never appear in the repository.

### How to add your models as secrets

1. Base64-encode each file:
   ```bash
   base64 -w 0 blaze_face_short_range.tflite > blaze_face_short_range.b64
   ```
2. Copy the entire content of the `.b64` file.
3. In your GitHub repository go to  
   **Settings → Secrets and variables → Actions → New repository secret**
4. Add three secrets:

| Secret name               | Content                                           |
|---------------------------|---------------------------------------------------|
| `BLAZE_FACE_MODEL`        | base64 of `blaze_face_short_range.tflite`         |
| `FACE_MESH_MODEL`         | base64 of `facemesh_lite_468_2022_09_06.f16.tflite`|
| `FACE_SEGMENTATION_MODEL` | base64 of `face_segmentation_xenoformer_xs_2024_04_02.int8.tflite` |

GitHub Actions has a **64 KB limit per secret**. If a model exceeds this, split
it with `split -b 60000` and reassemble in the workflow step.

---

## Local development

Copy or symlink the model files into `app/src/main/assets/models/` before
running `./gradlew assembleDebug`. The folder is listed in `.gitignore` so
your local models are never accidentally committed.
