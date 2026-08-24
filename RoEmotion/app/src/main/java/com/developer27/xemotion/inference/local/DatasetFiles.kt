package com.developer27.xemotion.inference.local

import java.io.File
import java.util.Locale

internal object DatasetFiles {
    fun isImage(file: File): Boolean = file.isFile && isImageName(file.name)

    fun isImageName(name: String?): Boolean {
        val normalized = name?.lowercase(Locale.US) ?: return false
        return normalized.endsWith(".jpg") ||
            normalized.endsWith(".jpeg") ||
            normalized.endsWith(".png") ||
            normalized.endsWith(".bmp") ||
            normalized.endsWith(".webp")
    }
}
