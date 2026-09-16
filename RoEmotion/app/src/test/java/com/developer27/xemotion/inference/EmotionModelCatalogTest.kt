package com.developer27.xemotion.inference

import android.content.SharedPreferences
import com.developer27.xemotion.videoprocessing.Settings
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.lang.reflect.Proxy

class EmotionModelCatalogTest {
    @After
    fun restoreTraceType() {
        Settings.Trace.updateInferenceType(Settings.Trace.DEFAULT_TYPE)
    }

    @Test
    fun existingInstallationsAndUnknownSelectionsUseTheOriginalModel() {
        assertEquals(EmotionModelCatalog.legacy, EmotionModelCatalog.selected(preferences()))
        assertEquals(
            EmotionModelCatalog.legacy,
            EmotionModelCatalog.selected(preferences(mapOf(EmotionModelCatalog.KEY_SELECTION to "retired")))
        )
        assertEquals(
            listOf("Angry", "Anxiety", "Disgust", "Excitement", "Sadness"),
            EmotionModelCatalog.legacy.labels
        )
        assertNull(EmotionModelCatalog.legacy.requiredTraceType)
    }

    @Test
    fun everyTrainingArchitectureAndTraceCombinationHasOneDistinctAsset() {
        val models = EmotionModelCatalog.models
        assertEquals(13, models.size)
        assertEquals(models.size, models.map { it.id }.toSet().size)
        assertEquals(models.size, models.map { it.assetName }.toSet().size)
        for (distribution in EmotionTrainingDistribution.entries) {
            for (architecture in EmotionModelArchitecture.entries) {
                for (trace in EmotionTraceVariant.entries) {
                    val model = models.single {
                        it.trainingDistribution == distribution &&
                            it.architecture == architecture && it.traceVariant == trace
                    }
                    assertEquals(
                        "current_${distribution.assetKey}_${architecture.assetKey}_${trace.assetKey}.pt",
                        model.assetName
                    )
                    assertEquals(trace.traceType, model.requiredTraceType)
                }
            }
        }
    }

    @Test
    fun currentModelsPreserveTrainingClassOrderAndSequenceLength() {
        val expectedLabels = listOf(
            "Amusement", "Anger", "Anxiety", "Disgust", "Fear",
            "Joy", "Neutral", "Sadness", "Surprise"
        )
        EmotionModelCatalog.models.filterNot { it.isLegacy }.forEach { model ->
            assertEquals(expectedLabels, model.labels)
            assertEquals(
                if (model.architecture == EmotionModelArchitecture.STANDALONE) 1 else 5,
                model.sequenceLength
            )
            assertTrue(model.selectionSummary.contains(model.trainingDistribution!!.displayName))
            assertTrue(model.selectionSummary.contains(model.traceVariant!!.displayName))
        }
    }

    @Test
    fun finalSelectionPersistsTheModelAndMatchingTraceTogether() {
        val preferences = preferences()
        EmotionModelCatalog.models.filterNot { it.isLegacy }.forEach { model ->
            EmotionModelCatalog.select(preferences, model)

            assertEquals(model, EmotionModelCatalog.selected(preferences))
            assertEquals(
                model.requiredTraceType!!.name,
                preferences.getString(Settings.KEY_INFERENCE_TRACE_TYPE, null)
            )
            assertEquals(model.requiredTraceType, Settings.Trace.inferenceType)
        }
    }

    @Test
    fun selectingPreviousModelPreservesTheUserControlledTraceSetting() {
        val preferences = preferences(mapOf(Settings.KEY_INFERENCE_TRACE_TYPE to "RAW_CV"))
        Settings.Trace.updateInferenceType(Settings.Trace.Type.RAW_CV)

        EmotionModelCatalog.select(preferences, EmotionModelCatalog.legacy)

        assertEquals(EmotionModelCatalog.legacy, EmotionModelCatalog.selected(preferences))
        assertEquals("RAW_CV", preferences.getString(Settings.KEY_INFERENCE_TRACE_TYPE, null))
        assertEquals(Settings.Trace.Type.RAW_CV, Settings.Trace.inferenceType)
    }

    @Test(expected = IllegalArgumentException::class)
    fun aModifiedOrUnknownModelCannotBePersisted() {
        EmotionModelCatalog.select(
            preferences(),
            EmotionModelCatalog.legacy.copy(assetName = "unverified.pt")
        )
    }

    /** Minimal in-memory Android preferences for testing selection without a device. */
    private fun preferences(initial: Map<String, String> = emptyMap()): SharedPreferences {
        val values = initial.toMutableMap()
        return Proxy.newProxyInstance(
            SharedPreferences::class.java.classLoader,
            arrayOf(SharedPreferences::class.java)
        ) { _, method, arguments ->
            when (method.name) {
                "getString" -> values[arguments!![0] as String] ?: arguments[1]
                "edit" -> {
                    val pending = mutableMapOf<String, String>()
                    Proxy.newProxyInstance(
                        SharedPreferences.Editor::class.java.classLoader,
                        arrayOf(SharedPreferences.Editor::class.java)
                    ) { editor, editorMethod, editorArguments ->
                        when (editorMethod.name) {
                            "putString" -> {
                                pending[editorArguments!![0] as String] = editorArguments[1] as String
                                editor
                            }
                            "apply" -> {
                                values.putAll(pending)
                                null
                            }
                            else -> error("Unexpected editor method: ${editorMethod.name}")
                        }
                    }
                }
                else -> error("Unexpected preferences method: ${method.name}")
            }
        } as SharedPreferences
    }
}
