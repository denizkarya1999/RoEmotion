package com.developer27.xemotion.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

object PyTorchModuleLoader {
    private const val TAG = "PyTorchModuleLoader"

    /**
     * Copies `<assetName>` from assets → internal storage if needed,
     * then loads it with PyTorch Mobile.
     */
    fun loadFromAssets(context: Context, assetName: String): Module {
        val outFile = File(context.filesDir, assetName)
        // overwrite every time
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile, false).use { output ->
                input.copyTo(output)
            }
        }
        if (!outFile.exists() || outFile.length() == 0L) {
            throw RuntimeException("Failed to copy model from assets")
        }
        // optional debug header check:
        val header = ByteArray(2)
        FileInputStream(outFile).use { it.read(header) }
        Log.i("PTLoader","Header bytes: ${header.joinToString{ "%02x".format(it) }}")
        return Module.load(outFile.absolutePath)
    }
}

class PyTorchClassifier private constructor(
    private val module: Module,

    // 224x224 Center Cropping
    private val inputWidth: Int = 224,
    private val inputHeight: Int = 224,

    // Applying ImageNet Normalizations
    private val mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
    private val std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)
) : Closeable {

    companion object {
        private const val TAG = "PyTorchClassifier"
        @Volatile private var instance: PyTorchClassifier? = null

        // Hardcoded emotion labels
        private val labels = listOf("Angry", "Anxiety", "Disgust", "Excitement", "Sadness")

        /**
         * Load the .pt model from assets and return (or cache) the classifier.
         * Call: PyTorchClassifier.fromAsset(context, "RoEmotion_Emotion_Detection_ResNet_50_LSTM_Attention.pt")
         */
        fun fromAsset(
            context: Context,
            modelAsset: String = "RoEmotion_Emotion_Detection_ResNet_50_LSTM_Attention.pt"
        ): PyTorchClassifier {
            return instance ?: synchronized(this) {
                val module = PyTorchModuleLoader.loadFromAssets(context, modelAsset)
                PyTorchClassifier(module).also { instance = it }
            }
        }
    }

    /**
     * Runs the model on the given bitmap and returns:
     *  • bestLabel: highest probability class (after softmax)
     *  • probs: FloatArray of softmax probabilities
     */
    fun classifySequence(frames: List<Bitmap>): Pair<String, FloatArray> {
        // 1) Preprocess each frame → FloatArray
        val singleNumel = 3 * inputHeight * inputWidth
        val seqLen      = frames.size
        val seqFloats   = FloatArray(singleNumel * seqLen)

        for ((i, bmp) in frames.withIndex()) {
            // Replace direct resize with training-consistent preprocessing
            val processed = preprocess(bmp)

            val flat = TensorImageUtils.bitmapToFloat32Tensor(processed, mean, std).dataAsFloatArray
            System.arraycopy(flat, 0, seqFloats, i * singleNumel, singleNumel)
        }

        // 2) Build a [1,5,3,224,224] tensor
        val seqTensor = org.pytorch.Tensor.fromBlob(
            seqFloats,
            longArrayOf(1, seqLen.toLong(), 3, inputHeight.toLong(), inputWidth.toLong())
        )

        // 3) Forward
        val outputTensor = module.forward(IValue.from(seqTensor)).toTensor()
        val rawLogits    = outputTensor.dataAsFloatArray

        // 4) Softmax + argmax (as you already do)
        val expScores = rawLogits.map { Math.exp(it.toDouble()).toFloat() }
        val sumExp    = expScores.sum()
        val probs     = expScores.map { it / sumExp }.toFloatArray()
        val maxIndex  = probs.indices.maxByOrNull { probs[it] } ?: 0
        val bestLabel = labels.getOrElse(maxIndex) { "Unknown" }
        return bestLabel to probs
    }

    // Full preprocessing pipeline to match training exactly
    private fun preprocess(bitmap: Bitmap): Bitmap {
        val resized = resizeShortSide(bitmap, 256) // Resize shorter side to 256
        return centerCrop(resized, inputWidth, inputHeight) // Then center crop to 224x224
    }

    // Resize while preserving aspect ratio (matches transforms.Resize(256))
    private fun resizeShortSide(bitmap: Bitmap, shortSide: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w <= 0 || h <= 0) return bitmap

        val scale = if (w < h) {
            shortSide.toFloat() / w.toFloat()
        } else {
            shortSide.toFloat() / h.toFloat()
        }

        val newW = (w * scale).toInt().coerceAtLeast(1)
        val newH = (h * scale).toInt().coerceAtLeast(1)

        return Bitmap.createScaledBitmap(bitmap, newW, newH, true)
    }

    // Center crop (matches transforms.CenterCrop(224))
    private fun centerCrop(bitmap: Bitmap, cropWidth: Int, cropHeight: Int): Bitmap {
        val x = ((bitmap.width - cropWidth) / 2).coerceAtLeast(0)
        val y = ((bitmap.height - cropHeight) / 2).coerceAtLeast(0)

        val w = cropWidth.coerceAtMost(bitmap.width)
        val h = cropHeight.coerceAtMost(bitmap.height)

        return Bitmap.createBitmap(bitmap, x, y, w, h)
    }

    // Log predictions for each class
    fun classifyAndLogSequence(frames: List<Bitmap>) {
        val (_, probs) = classifySequence(frames)
        for (i in probs.indices) {
            Log.d(TAG, "${labels.getOrNull(i) ?: "Label$i"}: ${probs[i]}")
        }
    }

    // Turn off the model when finished
    override fun close() {
        module.destroy()
    }
}
