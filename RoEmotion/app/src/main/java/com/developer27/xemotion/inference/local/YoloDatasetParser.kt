package com.developer27.xemotion.inference.local

import java.io.File
import java.util.Locale

data class YoloDatasetSplit(
    val splitName: String,
    val imagesFolder: File,
    val labelsFolder: File?,
    val imageFiles: List<File>
)

/** Parses common Roboflow and classic YOLO dataset layouts. */
class YoloDatasetParser {
    fun readYaml(root: File): String? = findYaml(root)?.let { runCatching(it::readText).getOrNull() }

    fun parseClassNames(yaml: String): List<String> {
        val lines = yaml.lines()
        val inline = Regex("""^\s*names\s*:\s*\[(.*)]\s*$""")
        lines.forEach { line ->
            inline.find(line)?.let { match ->
                return match.groupValues[1]
                    .split(",")
                    .map { it.trim().trim('"', '\'') }
                    .filter(String::isNotEmpty)
            }
        }

        val start = lines.indexOfFirst { it.trim().startsWith("names:") }
        if (start < 0) return emptyList()
        val block = lines.drop(start + 1)
            .takeWhile { it.isBlank() || it.firstOrNull()?.isWhitespace() == true }
            .filter(String::isNotBlank)

        val mapped = block.mapNotNull { line ->
            Regex("""^\s*\d+\s*:\s*(.+?)\s*$""")
                .find(line)
                ?.groupValues
                ?.get(1)
                ?.trim()
                ?.trim('"', '\'')
        }
        if (mapped.isNotEmpty()) return mapped

        return block.mapNotNull { line ->
            Regex("""^\s*-\s*(.+?)\s*$""")
                .find(line)
                ?.groupValues
                ?.get(1)
                ?.trim()
                ?.trim('"', '\'')
        }
    }

    fun collectSplits(root: File, yaml: String): List<YoloDatasetSplit> {
        val configuredPaths = parseSplitPaths(yaml)
        val seenPaths = mutableSetOf<String>()
        return SPLIT_NAMES.mapNotNull { requestedName ->
            val directory = candidateDirectories(root, requestedName, configuredPaths[requestedName])
                .mapNotNull { runCatching { it.canonicalFile }.getOrNull() }
                .firstOrNull(File::isDirectory)
                ?: return@mapNotNull null
            if (!seenPaths.add(directory.canonicalPath)) return@mapNotNull null

            val images = directory.listFiles()
                ?.filter(DatasetFiles::isImage)
                ?.sortedBy { it.name.lowercase(Locale.US) }
                .orEmpty()
            if (images.isEmpty()) return@mapNotNull null

            YoloDatasetSplit(
                splitName = normalizedSplitName(requestedName, directory),
                imagesFolder = directory,
                labelsFolder = findLabelsDirectory(directory),
                imageFiles = images
            )
        }
    }

    fun readGroundTruthClassIds(labelsFolder: File?, image: File): List<Int> {
        if (labelsFolder?.isDirectory != true) return emptyList()
        val labelFile = File(labelsFolder, "${image.nameWithoutExtension}.txt")
        return runCatching(labelFile::readLines).getOrDefault(emptyList()).mapNotNull { line ->
            line.trim().split(Regex("""\s+""")).firstOrNull()?.toIntOrNull()
        }
    }

    private fun findYaml(root: File): File? =
        listOf(File(root, "data.yaml"), File(root, "dataset.yaml")).firstOrNull(File::isFile)
            ?: root.listFiles()?.firstOrNull {
                it.isFile && (it.extension.equals("yaml", true) || it.extension.equals("yml", true))
            }

    private fun parseSplitPaths(yaml: String): Map<String, String> = buildMap {
        val pattern = Regex("""^\s*(train|val|test)\s*:\s*(.+?)\s*$""", RegexOption.IGNORE_CASE)
        yaml.lines().forEach { rawLine ->
            pattern.find(rawLine.substringBefore("#").trim())?.let { match ->
                put(
                    match.groupValues[1].lowercase(Locale.US),
                    match.groupValues[2].trim().trim('"', '\'')
                )
            }
        }
    }

    private fun candidateDirectories(root: File, split: String, configured: String?): List<File> {
        val result = mutableListOf<File>()
        configured?.let { path ->
            val normalized = path.replace("\\", "/").removePrefix("./").removePrefix("../")
            result += File(root, normalized)
            root.parentFile?.let { result += File(it, normalized) }
        }
        when (split) {
            "train" -> result += listOf(File(root, "train/images"), File(root, "images/train"))
            "val", "valid", "validation" -> result += listOf(
                File(root, "valid/images"), File(root, "val/images"),
                File(root, "validation/images"), File(root, "images/valid"),
                File(root, "images/val"), File(root, "images/validation")
            )
            "test", "testing" -> result += listOf(
                File(root, "test/images"), File(root, "testing/images"),
                File(root, "images/test"), File(root, "images/testing")
            )
        }
        return result
    }

    private fun findLabelsDirectory(images: File): File? {
        val sibling = File(images.parentFile ?: return null, "labels")
        return sibling.takeIf(File::isDirectory)
    }

    private fun normalizedSplitName(requested: String, images: File): String {
        val names = listOf(images.name, images.parentFile?.name.orEmpty(), requested)
            .map { it.lowercase(Locale.US) }
        return when {
            names.any { it == "valid" || it == "validation" } -> "valid"
            names.any { it == "val" } -> "val"
            names.any { it == "test" || it == "testing" } -> "test"
            else -> "train"
        }
    }

    private companion object {
        val SPLIT_NAMES = listOf("train", "val", "valid", "validation", "test", "testing")
    }
}
