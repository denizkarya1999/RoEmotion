package com.developer27.xemotion.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class EmotionSequenceSamplerTest {
    @Test
    fun currentModelsKeepShortRealSequencesWithoutInventingFrames() {
        assertEquals(listOf("first"), sampleEmotionSequence(listOf("first"), 5, false))
        assertEquals(listOf(1, 2, 3), sampleEmotionSequence(listOf(1, 2, 3), 5, false))
    }

    @Test
    fun legacyModelStillReceivesItsRequiredFiveFrames() {
        assertEquals(listOf(1, 2, 2, 2, 2), sampleEmotionSequence(listOf(1, 2), 5, true))
    }

    @Test
    fun samplingLongSequencePreservesChronologyAndBothEndpoints() {
        assertEquals(listOf(1, 3, 5, 7, 9), sampleEmotionSequence((1..9).toList(), 5, false))
    }

    @Test
    fun standaloneReceivesOnlyTheFirstSelectedImage() {
        assertEquals(listOf(10), sampleEmotionSequence(listOf(10, 20, 30), 1, false))
    }

    @Test(expected = IllegalArgumentException::class)
    fun emptySelectionCannotBecomeAnInferenceSequence() {
        sampleEmotionSequence(emptyList<Int>(), 5, false)
    }
}
