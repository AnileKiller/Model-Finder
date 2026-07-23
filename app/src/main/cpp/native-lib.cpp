#include <jni.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Internal C++ beauty pipeline (implemented elsewhere in the engine).
// Pixels are ARGB_8888 packed ints; masks are row-major float[w*h] in [0,1].
// Each function operates on `pixels` in place.
// ---------------------------------------------------------------------------
void applyBlemishReduction(int32_t* pixels, const float* blemishMask,
                           int w, int h, float strength);
void applyFaceSharpening(int32_t* pixels, const float* sharpMask,
                         int w, int h, float strength);
void applyEyebrowDefinition(int32_t* pixels, const float* eyebrowMask,
                            int w, int h, float strength);
void applySkinBrightness(int32_t* pixels, const float* brightnessMask,
                         int w, int h, float strength);
void applyUnderEyeReduction(int32_t* pixels, const float* eyeBagsMask,
                            const float* eyesMask, const float* refinedMask,
                            int w, int h, float strength);
void applyEyeBrightness(int32_t* pixels, const float* irisMask,
                        int w, int h, float strength);
void applyTeethWhitening(int32_t* pixels, const float* teethMask,
                         int w, int h, float strength);

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_beautycamera_nativebridge_NativeBeautyEngine_processImage(
        JNIEnv* env,
        jobject /* this: NativeBeautyEngine object */,
        jintArray jPixels,
        jint width,
        jint height,
        jfloatArray jRefinedMask,
        jfloatArray jSharpMask,
        jfloatArray jBlemishMask,
        jfloatArray jBrightnessMask,
        jfloatArray jEyebrowMask,
        jfloatArray jEyesMask,
        jfloatArray jEyeBagsMask,
        jfloatArray jIrisMask,
        jfloatArray jTeethMask,
        jfloat blemishStrength,
        jfloat sharpenStrength,
        jfloat eyebrowStrength,
        jfloat skinBrightnessStrength,
        jfloat underEyeStrength,
        jfloat eyeBrightnessStrength,
        jfloat teethWhiteningStrength) {

    const jsize pixelCount = env->GetArrayLength(jPixels);
    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);

    // -----------------------------------------------------------------------
    // We never mutate the caller's array. Instead we allocate a fresh output
    // array, copy the source pixels into it, process in place, and return it.
    // -----------------------------------------------------------------------
    jintArray jResult = env->NewIntArray(pixelCount);
    if (jResult == nullptr) {
        return nullptr; // OutOfMemoryError already thrown
    }

    // Acquire pointers -------------------------------------------------------
    jint*   srcPixels      = env->GetIntArrayElements(jPixels, nullptr);

    jfloat* refinedMask    = env->GetFloatArrayElements(jRefinedMask, nullptr);
    jfloat* sharpMask      = env->GetFloatArrayElements(jSharpMask, nullptr);
    jfloat* blemishMask    = env->GetFloatArrayElements(jBlemishMask, nullptr);
    jfloat* brightnessMask = env->GetFloatArrayElements(jBrightnessMask, nullptr);
    jfloat* eyebrowMask    = env->GetFloatArrayElements(jEyebrowMask, nullptr);
    jfloat* eyesMask       = env->GetFloatArrayElements(jEyesMask, nullptr);
    jfloat* eyeBagsMask    = env->GetFloatArrayElements(jEyeBagsMask, nullptr);
    jfloat* irisMask       = env->GetFloatArrayElements(jIrisMask, nullptr);
    jfloat* teethMask      = env->GetFloatArrayElements(jTeethMask, nullptr);

    // Working buffer: copy of the source pixels inside the result array.
    jint* resultPixels = env->GetIntArrayElements(jResult, nullptr);
    for (jsize i = 0; i < pixelCount; ++i) {
        resultPixels[i] = srcPixels[i];
    }

    int32_t* pixels = reinterpret_cast<int32_t*>(resultPixels);

    // Pipeline — same order as ApplyBeautyUseCase.kt --------------------------
    if (blemishStrength > 0.0f) {
        applyBlemishReduction(pixels, blemishMask, w, h, blemishStrength);
    }
    if (sharpenStrength > 0.0f) {
        applyFaceSharpening(pixels, sharpMask, w, h, sharpenStrength);
    }
    if (eyebrowStrength > 0.0f) {
        applyEyebrowDefinition(pixels, eyebrowMask, w, h, eyebrowStrength);
    }
    if (skinBrightnessStrength > 0.0f) {
        applySkinBrightness(pixels, brightnessMask, w, h, skinBrightnessStrength);
    }
    if (underEyeStrength > 0.0f) {
        applyUnderEyeReduction(pixels, eyeBagsMask, eyesMask, refinedMask,
                               w, h, underEyeStrength);
    }
    if (eyeBrightnessStrength > 0.0f) {
        applyEyeBrightness(pixels, irisMask, w, h, eyeBrightnessStrength);
    }
    if (teethWhiteningStrength > 0.0f) {
        applyTeethWhitening(pixels, teethMask, w, h, teethWhiteningStrength);
    }

    // Release ----------------------------------------------------------------
    // Result buffer: mode 0 => copy changes back to the Java array AND free
    // the native buffer. This is what makes the processed pixels visible to
    // Kotlin.
    env->ReleaseIntArrayElements(jResult, resultPixels, 0);

    // Source pixels were never modified: JNI_ABORT => free without copy-back.
    env->ReleaseIntArrayElements(jPixels, srcPixels, JNI_ABORT);

    // Masks are read-only: JNI_ABORT => free the buffers, no copy-back.
    env->ReleaseFloatArrayElements(jRefinedMask,    refinedMask,    JNI_ABORT);
    env->ReleaseFloatArrayElements(jSharpMask,      sharpMask,      JNI_ABORT);
    env->ReleaseFloatArrayElements(jBlemishMask,    blemishMask,    JNI_ABORT);
    env->ReleaseFloatArrayElements(jBrightnessMask, brightnessMask, JNI_ABORT);
    env->ReleaseFloatArrayElements(jEyebrowMask,    eyebrowMask,    JNI_ABORT);
    env->ReleaseFloatArrayElements(jEyesMask,       eyesMask,       JNI_ABORT);
    env->ReleaseFloatArrayElements(jEyeBagsMask,    eyeBagsMask,    JNI_ABORT);
    env->ReleaseFloatArrayElements(jIrisMask,       irisMask,       JNI_ABORT);
    env->ReleaseFloatArrayElements(jTeethMask,      teethMask,      JNI_ABORT);

    return jResult;
}
