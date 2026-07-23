plugins {
    id("com.android.library")
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.beautyai.prototype"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        compose = true
    }

    // Keep model files uncompressed in the APK so they can be memory-mapped.
    // "task" covers MediaPipe Tasks bundles; "tflite"/"lite" cover raw TFLite models.
    androidResources {
        noCompress += listOf("tflite", "lite", "task")
    }

    packaging {
        // libc++_shared.so is shipped by both the MediaPipe Tasks AAR and the
        // TFLite GPU AAR. pickFirsts silences the "duplicate file" build error
        // by keeping whichever version is resolved first (both are ABI-compatible).
        jniLibs {
            pickFirsts += listOf("**/libc++_shared.so")
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.material.icons.extended)
    debugImplementation(libs.androidx.ui.tooling)

    implementation(libs.coroutines.android)

    // TensorFlow Lite — core runtime
    implementation(libs.tensorflow.lite)
    implementation(libs.tensorflow.lite.support)

    // GPU delegate for hardware acceleration (NNAPI is bundled in the core runtime).
    // NOTE: only the legacy direct GpuDelegate/CompatibilityList API is used here —
    // do NOT also add tensorflow-lite-gpu-delegate-plugin (Play Services variant),
    // it ships a conflicting definition of GpuDelegateFactory.Options that breaks
    // compilation ("Cannot access class ... conflicting dependencies").
    implementation(libs.tensorflow.lite.gpu)
    // GpuDelegateFactory.Options (used via CompatibilityList.bestOptionsForThisDevice)
    // lives in this separate API artifact. tensorflow-lite-gpu only pulls it in
    // transitively as an implementation dependency, which is NOT visible on our
    // compile classpath — it must be declared directly here.
    implementation(libs.tensorflow.lite.gpu.api)

    // MediaPipe Tasks — provides FaceLandmarker which runs face detection +
    // landmark extraction in a single call, and ships the custom C++ ops
    // (e.g. Landmarks2TransformMatrix) that face_landmark_with_attention needs.
    // These ops are NOT available in the standard TFLite runtime.
    implementation(libs.mediapipe.tasks.vision)

    // Image loading
    implementation(libs.coil.compose)
}
