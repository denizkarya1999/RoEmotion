package com.developer27.xemotion.inference.local

import androidx.lifecycle.ViewModel

class LocalInferenceViewModel : ViewModel() {
    var pendingDatasetKind: DatasetKind? = null
    private val inputs = mutableMapOf<DatasetKind, DatasetInput>()

    fun setInput(kind: DatasetKind, input: DatasetInput) {
        inputs[kind] = input
    }

    fun inputFor(kind: DatasetKind): DatasetInput? = inputs[kind]
}
