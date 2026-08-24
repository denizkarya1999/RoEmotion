package com.developer27.xemotion.inference.local

import java.io.File
import java.util.Locale

data class ResNetSequence(
    val sequenceDir: File,
    val frameFiles: List<File>,
    val groundTruthLabel: String,
    val splitName: String,
    val relativePath: String
)

/** Discovers labeled image sequences without depending on Android UI classes. */
class ResNetDatasetParser {
    fun collectSequences(root: File): List<ResNetSequence> =
        root.walkTopDown()
            .filter(File::isDirectory)
            .mapNotNull { directory -> sequenceFromDirectory(root, directory) }
            .sortedBy { it.relativePath.lowercase(Locale.US) }
            .toList()

    private fun sequenceFromDirectory(root: File, directory: File): ResNetSequence? {
        val frames = directory.listFiles()
            ?.filter(DatasetFiles::isImage)
            ?.sortedBy { it.name.lowercase(Locale.US) }
            .orEmpty()
        if (frames.isEmpty()) return null

        val relativePath = directory.relativeTo(root).invariantSeparatorsPath
        val parts = relativePath.split("/").filter(String::isNotBlank)
        val splitIndex = parts.indexOfFirst { it.lowercase(Locale.US) in SPLIT_NAMES }
        val splitName = parts.getOrNull(splitIndex)?.lowercase(Locale.US) ?: UNKNOWN_SPLIT
        val label = when {
            splitIndex >= 0 && parts.size > splitIndex + 1 -> parts[splitIndex + 1]
            parts.size >= 2 -> parts[parts.lastIndex - 1]
            parts.isNotEmpty() -> parts.first()
            else -> directory.name
        }
        return ResNetSequence(directory, frames, label, splitName, relativePath)
    }

    private companion object {
        const val UNKNOWN_SPLIT = "unknown"
        val SPLIT_NAMES = setOf("train", "val", "valid", "validation", "test", "testing")
    }
}
