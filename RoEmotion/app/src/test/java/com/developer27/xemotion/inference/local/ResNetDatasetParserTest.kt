package com.developer27.xemotion.inference.local

import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ResNetDatasetParserTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun collectSequences_extractsSplitLabelAndSortedFrames() {
        val root = temporaryFolder.newFolder("dataset")
        val sequence = File(root, "train/Excitement/sequence-01").apply { mkdirs() }
        File(sequence, "frame-10.png").writeText("image")
        File(sequence, "frame-02.jpg").writeText("image")
        File(sequence, "notes.txt").writeText("ignored")

        val result = ResNetDatasetParser().collectSequences(root)

        assertEquals(1, result.size)
        with(result.single()) {
            assertEquals("train", splitName)
            assertEquals("Excitement", groundTruthLabel)
            assertEquals("train/Excitement/sequence-01", relativePath)
            assertEquals(listOf("frame-02.jpg", "frame-10.png"), frameFiles.map(File::getName))
        }
    }

    @Test
    fun collectSequences_supportsDatasetsWithoutNamedSplit() {
        val root = temporaryFolder.newFolder("dataset-without-split")
        val sequence = File(root, "Sadness/sequence-02").apply { mkdirs() }
        File(sequence, "frame.jpg").writeText("image")

        val result = ResNetDatasetParser().collectSequences(root).single()

        assertEquals("unknown", result.splitName)
        assertEquals("Sadness", result.groundTruthLabel)
    }
}
