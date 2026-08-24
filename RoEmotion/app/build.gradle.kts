plugins {
    id("com.android.application")
}

abstract class PrepareModernCxxRuntime : org.gradle.api.tasks.Sync() {
    @get:org.gradle.api.tasks.OutputDirectory
    abstract val outputDirectory: org.gradle.api.file.DirectoryProperty

    init {
        into(outputDirectory)
    }
}

val generatedCxxRuntime = layout.buildDirectory.dir("generated/modernCxxRuntime/jniLibs")
val prepareModernCxxRuntime by tasks.registering(PrepareModernCxxRuntime::class) {
    outputDirectory.set(generatedCxxRuntime)
    val ndkLibraryRoot = androidComponents.sdkComponents.ndkDirectory.map { ndk ->
        val prebuiltRoot = ndk.dir("toolchains/llvm/prebuilt").asFile
        val hostToolchain = checkNotNull(
            prebuiltRoot.listFiles()?.singleOrNull(File::isDirectory)
        ) { "Expected one NDK host toolchain under ${prebuiltRoot.absolutePath}" }
        hostToolchain.resolve("sysroot/usr/lib")
    }
    into(generatedCxxRuntime)
    from(ndkLibraryRoot.map { it.resolve("aarch64-linux-android/libc++_shared.so") }) {
        into("arm64-v8a")
    }
    from(ndkLibraryRoot.map { it.resolve("arm-linux-androideabi/libc++_shared.so") }) {
        into("armeabi-v7a")
    }
    from(ndkLibraryRoot.map { it.resolve("x86_64-linux-android/libc++_shared.so") }) {
        into("x86_64")
    }
}

android {
    namespace = "com.developer27.xemotion"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.developer27.xemotion"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86_64")
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        viewBinding = true
    }

    packaging {
        jniLibs {
            pickFirsts.add("lib/x86/libc++_shared.so")
            pickFirsts.add("lib/x86_64/libc++_shared.so")
            pickFirsts.add("lib/armeabi-v7a/libc++_shared.so")
            pickFirsts.add("lib/arm64-v8a/libc++_shared.so")
        }
    }
    androidResources {
        noCompress += listOf("tflite")
    }
}

androidComponents.onVariants { variant ->
    variant.sources.jniLibs?.addGeneratedSourceDirectory(
        prepareModernCxxRuntime,
        PrepareModernCxxRuntime::outputDirectory
    )
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    // Official OpenCV Android runtime. Version 5 is built for current Android toolchains.
    implementation("org.opencv:opencv:5.0.0.1")

    // PyTorch
    implementation("org.pytorch:pytorch_android:1.13.1") {
        exclude(group = "org.bytedeco", module = "libc++_shared")
    }

    // Current LiteRT CompiledModel runtime, including GPU acceleration.
    implementation("com.google.ai.edge.litert:litert:2.2.0")

    // Kotlin & Android core libs
    implementation("androidx.activity:activity-ktx:1.13.0")
    implementation("androidx.core:core-ktx:1.18.0")
    implementation("androidx.appcompat:appcompat:1.8.0")
    implementation("com.google.android.material:material:1.14.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.2")
    implementation("androidx.documentfile:documentfile:1.1.0")

    // Preferences
    implementation("androidx.preference:preference-ktx:1.2.1")

    //Splash screen
    implementation("androidx.core:core-splashscreen:1.2.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")

    // Apache Commons Math
    implementation("org.apache.commons:commons-math3:3.6.1")
}
