package com.developer27.xemotion.inference.local

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.util.Locale

/** Evaluates sequence classifications independently from Android UI and model loading. */
class ResNetDatasetEvaluator(
    private val parser: ResNetDatasetParser,
    private val sequenceLength: Int = 5
) {
    init {
        require(sequenceLength > 0) { "Sequence length must be positive." }
    }

    suspend fun evaluate(
        root: File,
        inputDescription: String,
        predictor: (List<Bitmap>) -> Pair<String, FloatArray>,
        onProgress: suspend (Int, Int, String) -> Unit
    ): String {
        val sequences = parser.collectSequences(root)
        require(sequences.isNotEmpty()) { "No valid ResNet image sequences were found." }
        val classNames = sequences.map { it.groundTruthLabel }.distinct().sorted()
        val classIndices = classNames.withIndex().associate { it.value to it.index }
        val groundTruthCounts = IntArray(classNames.size)
        val correctCounts = IntArray(classNames.size)
        val splitStats = linkedMapOf<String, Counts>()
        var overallCorrect = 0
        var evaluatedSequences = 0

        return buildString {
            appendLine("RoEmotion - ResNet Dataset Inference Report")
            appendLine("Selected Input: $inputDescription")
            appendLine("Detected Dataset Root: ${root.absolutePath}")
            appendLine("Detected Classes: ${classNames.joinToString()}")
            appendLine("Total Sequences: ${sequences.size}")
            appendLine()

            sequences.forEachIndexed { index, sequence ->
                onProgress(index + 1, sequences.size, "ResNet inferring sequences")
                val loaded = sequence.frameFiles.mapNotNull { BitmapFactory.decodeFile(it.absolutePath) }
                if (loaded.isEmpty()) {
                    appendLine("[Skipped] ${sequence.relativePath}")
                    return@forEachIndexed
                }

                try {
                    evaluatedSequences++
                    val prepared = adjustSequenceLength(loaded)
                    val (predictedLabel, probabilities) = predictor(prepared)
                    val success = normalize(sequence.groundTruthLabel) == normalize(predictedLabel)
                    classIndices[sequence.groundTruthLabel]?.let { classIndex ->
                        groundTruthCounts[classIndex]++
                        if (success) correctCounts[classIndex]++
                    }
                    val counts = splitStats.getOrPut(sequence.splitName, ::Counts)
                    counts.total++
                    if (success) {
                        counts.correct++
                        overallCorrect++
                    }

                    appendLine(sequence.relativePath)
                    appendLine("  Split: ${sequence.splitName}")
                    appendLine("  Ground Truth: ${sequence.groundTruthLabel}")
                    appendLine("  Prediction: $predictedLabel")
                    appendLine("  Frames Found: ${sequence.frameFiles.size}")
                    appendLine("  Frames Used: ${prepared.size}")
                    appendLine("  Success: ${if (success) "Yes" else "No"}")
                    appendLine("  Probabilities: ${formatProbabilities(probabilities)}")
                    appendLine()
                } finally {
                    loaded.forEach(Bitmap::recycle)
                }
            }

            appendLine("========== Per-Class Summary ==========")
            classNames.indices.forEach { index ->
                appendLine(
                    "${classNames[index]} -> Ground Truth: ${groundTruthCounts[index]}, " +
                        "Successful: ${correctCounts[index]}, Percentage: ${percentage(correctCounts[index], groundTruthCounts[index])}%"
                )
            }
            appendLine()
            appendLine("========== Split Summary ==========")
            splitStats.forEach { (name, counts) ->
                appendLine("$name -> Ground Truth: ${counts.total}, Successful: ${counts.correct}, Percentage: ${percentage(counts.correct, counts.total)}%")
            }
            appendLine()
            appendLine("========== Overall Summary ==========")
            appendLine("Overall Ground Truth Count: $evaluatedSequences")
            appendLine("Overall Successful Count: $overallCorrect")
            appendLine("Overall Success Percentage: ${percentage(overallCorrect, evaluatedSequences)}%")
        }
    }

    private fun adjustSequenceLength(frames: List<Bitmap>): List<Bitmap> = when {
        frames.size == sequenceLength -> frames
        frames.size > sequenceLength && sequenceLength == 1 -> listOf(frames.first())
        frames.size > sequenceLength -> List(sequenceLength) { index ->
            frames[(index * (frames.lastIndex).toDouble() / (sequenceLength - 1)).toInt()]
        }
        else -> frames + List(sequenceLength - frames.size) { frames.last() }
    }

    private fun normalize(label: String): String = label.trim()
        .lowercase(Locale.US)
        .replace("-", " ")
        .replace("_", " ")
        .replace(Regex("""\s+"""), " ")

    private fun formatProbabilities(probabilities: FloatArray): String =
        probabilities.joinToString { String.format(Locale.US, "%.4f", it) }

    private fun percentage(numerator: Int, denominator: Int): String =
        String.format(
            Locale.US,
            "%.2f",
            if (denominator == 0) 0.0 else numerator * 100.0 / denominator
        )

    private data class Counts(var total: Int = 0, var correct: Int = 0)
}
