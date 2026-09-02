package com.developer27.xemotion.inference.local

import android.net.Uri
import androidx.lifecycle.ViewModel

/** Retains standalone-inference selections and results across configuration changes. */
class StandaloneInferenceViewModel : ViewModel() {
    var ledImageUris: List<Uri> = emptyList()
    var emotionImageUris: List<Uri> = emptyList()
    var ledGroundTruthClassId: Int? = null
    var emotionGroundTruth: String? = null
    var ledResults: String = ""
    var emotionResults: String = ""
}
