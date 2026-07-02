# TensorFlow Lite — keep all inference classes
-keep class org.tensorflow.** { *; }
-keep class com.google.flatbuffers.** { *; }

# Keep GPU delegate JNI entry points
-keep class org.tensorflow.lite.gpu.** { *; }

# Preserve model metadata classes used by TFLite Support Library
-keep class org.tensorflow.lite.support.** { *; }
