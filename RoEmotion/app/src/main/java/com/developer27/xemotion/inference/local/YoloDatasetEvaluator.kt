package com.developer27.xemotion.inference.local

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.developer27.xemotion.inference.YoloDet
import java.io.File
import java.util.Locale

/** Evaluates a YOLO dataset independently from activities and model loading. */
class YoloDatasetEvaluator(private val parser: YoloDatasetParser) {
    suspend fun evaluate(
        root: File,
        inputDescription: String,
        predictor: (Bitmap) -> List<YoloDet>,
        onProgress: suspend (Int, Int, String) -> Unit
    ): String {
        val yaml = parser.readYaml(root)
            ?: throw IllegalStateException("Could not read data.yaml / dataset.yaml.")
        val classNames = parser.parseClassNames(yaml)
        require(classNames.isNotEmpty()) { "Could not find class names in the dataset YAML." }
        val splits = parser.collectSplits(root, yaml)
        val totalImages = splits.sumOf { it.imageFiles.size }
        require(totalImages > 0) { "No YOLO images were found for train/val/test." }

        val perClassGroundTruth = IntArray(classNames.size)
        val perClassCorrect = IntArray(classNames.size)
        val splitStats = linkedMapOf<String, SplitStats>()
        var imageLevelCorrect = 0
        var processed = 0
        var evaluatedImages = 0

        return buildString {
            appendLine("RoEmotion - YOLO Dataset Inference Report")
            appendLine("Selected Input: $inputDescription")
            appendLine("Detected Dataset Root: ${root.absolutePath}")
            appendLine("Class Names: ${classNames.joinToString()}")
            appendLine()

            splits.forEach { split ->
                val stats = SplitStats(discoveredImageCount = split.imageFiles.size)
                appendLine("========== Split: ${split.splitName} ==========")

                split.imageFiles.forEach imageLoop@{ imageFile ->
                    processed++
                    onProgress(processed, totalImages, "YOLO inferring ${split.splitName}")
                    val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                    if (bitmap == null) {
                        stats.skippedImageCount++
                        appendLine("${imageFile.name} [Skipped: unreadable image]")
                        appendLine()
                        return@imageLoop
                    }

                    val groundTruth = parser.readGroundTruthClassIds(split.labelsFolder, imageFile)
                        .toSet()
                    groundTruth.forEach { classId ->
                        if (classId in perClassGroundTruth.indices) {
                            perClassGroundTruth[classId]++
                            stats.groundTruth++
                        }
                    }
                    val predicted = try {
                        predictor(bitmap).map(YoloDet::classId).toSet()
                    } finally {
                        bitmap.recycle()
                    }

                    evaluatedImages++
                    stats.evaluatedImageCount++
                    val fullMatch = groundTruth == predicted
                    groundTruth.filter { it in predicted }.forEach { classId ->
                        if (classId in perClassCorrect.indices) {
                            perClassCorrect[classId]++
                            stats.correct++
                        }
                    }
                    if (fullMatch) {
                        imageLevelCorrect++
                        stats.fullImageMatches++
                    }

                    appendLine(imageFile.name)
                    appendLine("  GT: ${classLabels(groundTruth, classNames)}")
                    appendLine("  Pred: ${classLabels(predicted, classNames)}")
                    appendLine()
                }
                splitStats[split.splitName] = stats
            }

            appendLine("========== Per-Class Summary ==========")
            classNames.indices.forEach { index ->
                appendLine(
                    "${classNames[index]} -> Ground Truth: ${perClassGroundTruth[index]}, " +
                        "Successful: ${perClassCorrect[index]}, Percentage: ${percentage(perClassCorrect[index], perClassGroundTruth[index])}%"
                )
            }

            appendLine()
            appendLine("========== Split Summary ==========")
            splitStats.forEach { (name, stats) ->
                appendLine("$name ->")
                appendLine("  Ground Truth Count: ${stats.groundTruth}")
                appendLine("  Successful Count: ${stats.correct}")
                appendLine("  Success Percentage: ${percentage(stats.correct, stats.groundTruth)}%")
                appendLine("  Images Discovered: ${stats.discoveredImageCount}")
                appendLine("  Images Evaluated: ${stats.evaluatedImageCount}")
                appendLine("  Images Skipped: ${stats.skippedImageCount}")
                appendLine("  Image-Level Full Match Count: ${stats.fullImageMatches} / ${stats.evaluatedImageCount}")
                appendLine("  Image-Level Full Match Percentage: ${percentage(stats.fullImageMatches, stats.evaluatedImageCount)}%")
            }

            appendLine()
            appendLine("========== Overall Summary ==========")
            appendLine("Overall Ground Truth Count: ${perClassGroundTruth.sum()}")
            appendLine("Overall Successful Count: ${perClassCorrect.sum()}")
            appendLine("Overall Success Percentage: ${percentage(perClassCorrect.sum(), perClassGroundTruth.sum())}%")
            appendLine("Images Discovered: $totalImages")
            appendLine("Images Evaluated: $evaluatedImages")
            appendLine("Images Skipped: ${totalImages - evaluatedImages}")
            appendLine("Image-Level Full Match Count: $imageLevelCorrect / $evaluatedImages")
            appendLine("Image-Level Full Match Percentage: ${percentage(imageLevelCorrect, evaluatedImages)}%")
        }
    }

    private fun classLabels(ids: Set<Int>, labels: List<String>): String =
        if (ids.isEmpty()) "[none]" else ids.joinToString { labels.getOrElse(it) { "Class_$it" } }

    private fun percentage(numerator: Int, denominator: Int): String =
        String.format(
            Locale.US,
            "%.2f",
            if (denominator == 0) 0.0 else numerator * 100.0 / denominator
        )

    private data class SplitStats(
        var groundTruth: Int = 0,
        var correct: Int = 0,
        var fullImageMatches: Int = 0,
        val discoveredImageCount: Int,
        var evaluatedImageCount: Int = 0,
        var skippedImageCount: Int = 0
    )
}
