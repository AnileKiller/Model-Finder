plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.beautyai.prototype"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.beautyai.prototype"
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

    // Keep TFLite model files from being compressed in the APK so they can
    // be loaded with MappedByteBuffer for zero-copy memory access.
    androidResources {
        noCompress += listOf("tflite", "lite")
    }

    packaging {
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

    // Image loading
    implementation(libs.coil.compose)
}
