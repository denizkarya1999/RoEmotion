package com.developer27.xemotion.inference.local

import android.net.Uri
import androidx.lifecycle.ViewModel

/** Retains standalone-inference selections and results across configuration changes. */
class StandaloneInferenceViewModel : ViewModel() {
    var ledImageUris: List<Uri> = emptyList()
    var emotionImageUris: List<Uri> = emptyList()
    var ledGroundTruthClassIds: Set<Int> = emptySet()
    var isLedGroundTruthConfigured: Boolean = false
    var emotionGroundTruth: String? = null
    var ledResults: String = ""
    var emotionResults: String = ""
}
