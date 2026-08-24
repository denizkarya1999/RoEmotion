package com.developer27.xemotion.storage

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.annotation.RequiresApi
import java.io.File
import java.io.FileOutputStream

/** Centralizes public Pictures/Documents writes and Android-version handling. */
class MediaStoreRepository(context: Context) {
    private val appContext = context.applicationContext
    private val resolver = appContext.contentResolver

    fun saveJpeg(bitmap: Bitmap, relativeDirectory: String, fileName: String): Boolean =
        runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                saveJpegScoped(bitmap, relativeDirectory, fileName)
            } else {
                saveJpegLegacy(bitmap, relativeDirectory, fileName)
            }
        }.onFailure { error ->
            Log.e(TAG, "Unable to save $fileName", error)
        }.isSuccess

    fun appendDocumentLine(fileName: String, line: String): Boolean =
        runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                appendDocumentLineScoped(fileName, line)
            } else {
                appendDocumentLineLegacy(fileName, line)
            }
        }.onFailure { error ->
            Log.e(TAG, "Unable to append $fileName", error)
        }.isSuccess

    fun writeDocument(fileName: String, content: String, subdirectory: String): Uri =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            writeDocumentScoped(fileName, content, subdirectory)
        } else {
            writeDocumentLegacy(fileName, content, subdirectory)
        }

    private fun saveJpegScoped(bitmap: Bitmap, directory: String, fileName: String) {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, JPEG_MIME_TYPE)
            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                "${Environment.DIRECTORY_PICTURES}/$directory"
            )
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = requireNotNull(
            resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        ) { "MediaStore insert failed" }

        try {
            requireNotNull(resolver.openOutputStream(uri)).use { output ->
                check(bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, output))
            }
            check(resolver.update(
                uri,
                ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) },
                null,
                null
            ) > 0) { "MediaStore could not publish $fileName" }
        } catch (error: Exception) {
            resolver.delete(uri, null, null)
            throw error
        }
    }

    @Suppress("DEPRECATION")
    private fun saveJpegLegacy(bitmap: Bitmap, directory: String, fileName: String) {
        val pictures = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val targetDirectory = File(pictures, directory).apply { mkdirs() }
        FileOutputStream(File(targetDirectory, fileName)).use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, output))
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun appendDocumentLineScoped(fileName: String, line: String) {
        val relativePath = Environment.DIRECTORY_DOCUMENTS
        val collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val existing = findDocument(collection, relativePath, fileName)
        val isNewDocument = existing == null
        val uri = existing ?: requireNotNull(
            resolver.insert(
                collection,
                ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.MIME_TYPE, TEXT_MIME_TYPE)
                    put(MediaStore.MediaColumns.RELATIVE_PATH, relativePath)
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
            )
        ) { "MediaStore document insert failed" }

        try {
            requireNotNull(resolver.openOutputStream(uri, "wa")).bufferedWriter().use { writer ->
                writer.appendLine(line)
            }
            if (isNewDocument) {
                check(
                    resolver.update(
                        uri,
                        ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) },
                        null,
                        null
                    ) > 0
                ) { "MediaStore could not publish $fileName" }
            }
        } catch (error: Exception) {
            if (isNewDocument) resolver.delete(uri, null, null)
            throw error
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun writeDocumentScoped(fileName: String, content: String, subdirectory: String): Uri {
        val collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val uri = requireNotNull(
            resolver.insert(
                collection,
                ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.MIME_TYPE, TEXT_MIME_TYPE)
                    put(
                        MediaStore.MediaColumns.RELATIVE_PATH,
                        "${Environment.DIRECTORY_DOCUMENTS}/$subdirectory"
                    )
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
            )
        ) { "MediaStore document insert failed" }
        try {
            requireNotNull(resolver.openOutputStream(uri)).bufferedWriter().use { it.write(content) }
            check(
                resolver.update(
                    uri,
                    ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) },
                    null,
                    null
                ) > 0
            ) { "MediaStore could not publish $fileName" }
            return uri
        } catch (error: Exception) {
            resolver.delete(uri, null, null)
            throw error
        }
    }

    private fun findDocument(collection: Uri, relativePath: String, fileName: String): Uri? {
        val projection = arrayOf(MediaStore.MediaColumns._ID)
        val selection =
            "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND ${MediaStore.MediaColumns.RELATIVE_PATH}=?"
        val arguments = arrayOf(fileName, "$relativePath/")
        resolver.query(collection, projection, selection, arguments, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                return Uri.withAppendedPath(collection, cursor.getLong(0).toString())
            }
        }
        return null
    }

    @Suppress("DEPRECATION")
    private fun appendDocumentLineLegacy(fileName: String, line: String) {
        val documents = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        documents.mkdirs()
        File(documents, fileName).appendText("$line\n")
    }

    @Suppress("DEPRECATION")
    private fun writeDocumentLegacy(fileName: String, content: String, subdirectory: String): Uri {
        val documents = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val directory = File(documents, subdirectory).apply { mkdirs() }
        val file = File(directory, fileName).apply { writeText(content) }
        return Uri.fromFile(file)
    }

    private companion object {
        const val TAG = "MediaStoreRepository"
        const val JPEG_MIME_TYPE = "image/jpeg"
        const val TEXT_MIME_TYPE = "text/plain"
        const val JPEG_QUALITY = 90
    }
}
