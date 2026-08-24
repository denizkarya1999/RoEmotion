package com.developer27.xemotion.inference.local

import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import java.util.zip.ZipInputStream

enum class DatasetInputType { FOLDER, ZIP }
enum class DatasetKind { YOLO, RESNET }

data class DatasetInput(val uri: Uri, val type: DatasetInputType)
data class PreparedDatasetRoot(
    val directory: File,
    val description: String,
    private val workingDirectory: File
) {
    fun cleanup() {
        workingDirectory.deleteRecursively()
    }
}

/** Copies a selected tree or safely extracts a ZIP into an isolated cache directory. */
class LocalDatasetInputResolver(context: Context) {
    private val appContext = context.applicationContext
    private val resolver = appContext.contentResolver

    suspend fun prepare(
        input: DatasetInput,
        kind: DatasetKind,
        onProgress: suspend (Int, Int, String) -> Unit
    ): PreparedDatasetRoot {
        val inputRoot = File(appContext.cacheDir, CACHE_DIRECTORY).apply { mkdirs() }
        val workDirectory = File(
            inputRoot,
            "${System.currentTimeMillis()}_${UUID.randomUUID()}"
        ).apply { mkdirs() }

        try {
            when (input.type) {
                DatasetInputType.ZIP -> {
                    onProgress(1, 4, "Extracting ZIP")
                    unzip(input.uri, workDirectory)
                }
                DatasetInputType.FOLDER -> {
                    onProgress(1, 4, "Copying folder")
                    copyDocumentTree(input.uri, workDirectory)
                }
            }

            onProgress(2, 4, "Locating dataset root")
            val root = when (kind) {
                DatasetKind.YOLO -> findYoloRoot(workDirectory)
                DatasetKind.RESNET -> findResNetRoot(workDirectory)
            } ?: throw IllegalStateException("Could not detect a valid dataset root.")

            return PreparedDatasetRoot(root, input.uri.toString(), workDirectory)
        } catch (error: Throwable) {
            workDirectory.deleteRecursively()
            throw error
        }
    }

    private fun unzip(uri: Uri, outputDirectory: File) {
        resolver.openInputStream(uri)?.use { input ->
            ZipInputStream(BufferedInputStream(input)).use { zip ->
                var entry = zip.nextEntry
                while (entry != null) {
                    val output = safeZipTarget(outputDirectory, entry.name)
                    if (entry.isDirectory) {
                        output.mkdirs()
                    } else {
                        output.parentFile?.mkdirs()
                        FileOutputStream(output).use(zip::copyTo)
                    }
                    zip.closeEntry()
                    entry = zip.nextEntry
                }
            }
        } ?: throw IllegalStateException("Failed to open ZIP file.")
    }

    private fun safeZipTarget(destination: File, entryName: String): File {
        val output = File(destination, entryName)
        val destinationPath = destination.canonicalPath
        val outputPath = output.canonicalPath
        require(outputPath == destinationPath || outputPath.startsWith("$destinationPath${File.separator}")) {
            "Blocked unsafe ZIP entry: $entryName"
        }
        return output
    }

    private fun copyDocumentTree(uri: Uri, destination: File) {
        val root = DocumentFile.fromTreeUri(appContext, uri)
            ?: throw IllegalStateException("Could not open selected folder.")
        copyDocument(root, destination)
    }

    private fun copyDocument(document: DocumentFile, destination: File) {
        if (document.isDirectory) {
            val directory = document.name
                ?.takeIf { it.isNotBlank() }
                ?.let { File(destination, it) }
                ?: destination
            directory.mkdirs()
            document.listFiles().forEach { copyDocument(it, directory) }
        } else if (document.isFile) {
            val output = File(destination, document.name ?: UNKNOWN_FILE_NAME)
            resolver.openInputStream(document.uri)?.use { input ->
                FileOutputStream(output).use(input::copyTo)
            }
        }
    }

    private fun findYoloRoot(start: File): File? =
        start.walkTopDown().firstOrNull(::isYoloRoot)

    private fun isYoloRoot(directory: File): Boolean {
        val hasYaml = directory.listFiles()?.any { file ->
            file.isFile && (file.extension.equals("yaml", true) || file.extension.equals("yml", true))
        } == true
        if (!hasYaml) return false

        val splitNames = listOf("train", "valid", "val", "test")
        return splitNames.any { File(directory, "$it/images").isDirectory } ||
            splitNames.any { File(directory, "images/$it").isDirectory }
    }

    private fun findResNetRoot(start: File): File? =
        start.walkTopDown()
            .filter { it.isDirectory }
            .firstOrNull { directory -> directory.walkTopDown().any(DatasetFiles::isImage) }

    private companion object {
        const val CACHE_DIRECTORY = "local_inference_inputs"
        const val UNKNOWN_FILE_NAME = "unknown_file"
    }
}
