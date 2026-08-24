package com.developer27.xemotion.inference.local

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class YoloDatasetParserTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private val parser = YoloDatasetParser()

    @Test
    fun parseClassNames_supportsInlineAndMappedYaml() {
        assertEquals(
            listOf("Angry", "Sad"),
            parser.parseClassNames("names: ['Angry', \"Sad\"]")
        )
        assertEquals(
            listOf("User One", "User Two"),
            parser.parseClassNames(
                """
                names:
                  0: User One
                  1: 'User Two'
                train: train/images
                """.trimIndent()
            )
        )
    }

    @Test
    fun collectSplits_usesYamlPathsAndPairsLabels() {
        val root = temporaryFolder.newFolder("dataset")
        val images = File(root, "images/train").apply { mkdirs() }
        val labels = File(root, "images/labels").apply { mkdirs() }
        File(images, "frame-b.png").writeText("image")
        val firstImage = File(images, "frame-a.jpg").apply { writeText("image") }
        File(images, "ignore.txt").writeText("not an image")
        File(labels, "frame-a.txt").writeText("2 0.5 0.5 0.2 0.2\ninvalid\n1 0.4 0.4 0.1 0.1")

        val splits = parser.collectSplits(root, "train: images/train")

        assertEquals(1, splits.size)
        assertEquals("train", splits.single().splitName)
        assertEquals(listOf("frame-a.jpg", "frame-b.png"), splits.single().imageFiles.map(File::getName))
        assertEquals(labels.canonicalFile, splits.single().labelsFolder?.canonicalFile)
        assertEquals(listOf(2, 1), parser.readGroundTruthClassIds(labels, firstImage))
        assertNull(parser.collectSplits(root, "test: missing/images").firstOrNull { it.splitName == "test" })
    }
}
