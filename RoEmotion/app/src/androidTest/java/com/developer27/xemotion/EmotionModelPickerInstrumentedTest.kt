package com.developer27.xemotion

import android.content.SharedPreferences
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceManager
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.RootMatchers.isDialog
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.developer27.xemotion.inference.EmotionModelCatalog
import com.developer27.xemotion.inference.EmotionModelSpec
import com.developer27.xemotion.videoprocessing.Settings
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/** Exercises the real Options screen without initializing camera or model runtimes. */
@RunWith(AndroidJUnit4::class)
class EmotionModelPickerInstrumentedTest {
    private lateinit var preferences: SharedPreferences
    private var originalPreferences = emptyMap<String, Any?>()
    private var scenario: ActivityScenario<SettingsActivity>? = null

    @Before
    fun openOptionsWithOriginalModel() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        preferences = PreferenceManager.getDefaultSharedPreferences(context)
        originalPreferences = preferences.all.mapValues { (_, value) ->
            if (value is Set<*>) value.toSet() else value
        }
        assertTrue(
            preferences.edit().clear()
                .putString(Settings.KEY_OPERATING_MODE, Settings.OperatingMode.Mode.INFERENCE.name)
                .commit()
        )
        scenario = ActivityScenario.launch(SettingsActivity::class.java)
        onView(withText(R.string.change_emotion_detection_model)).check(matches(isDisplayed()))
    }

    @After
    fun restoreOriginalPreferences() {
        try {
            scenario?.close()
        } finally {
            if (::preferences.isInitialized) {
                val editor = preferences.edit().clear()
                originalPreferences.forEach { (key, value) ->
                    when (value) {
                        is String -> editor.putString(key, value)
                        is Boolean -> editor.putBoolean(key, value)
                        is Int -> editor.putInt(key, value)
                        is Long -> editor.putLong(key, value)
                        is Float -> editor.putFloat(key, value)
                        is Set<*> -> editor.putStringSet(key, value.filterIsInstance<String>().toSet())
                        null -> editor.putString(key, null)
                        else -> error("Unexpected preference type: ${value.javaClass}")
                    }
                }
                assertTrue(editor.commit())
                InstrumentationRegistry.getInstrumentation().runOnMainSync {
                    Settings.load(preferences)
                }
            }
        }
    }

    @Test
    fun randomStandaloneRawSelectionPersistsAcrossActivityRecreation() {
        openPicker()
        choose("Current Models")
        choose("Trained with Random Distribution")
        choose("Standalone ResNet-50")
        choose("Raw")

        val expected = model("current_random_standalone_raw")
        assertSelection(expected)
        assertEquals(Settings.Trace.Type.RAW, Settings.Trace.inferenceType)

        requireNotNull(scenario).recreate()

        assertSelection(expected)
    }

    @Test
    fun personAttentionKalmanSelectionRetainsPendingChoicesThroughRotation() {
        openPicker()
        choose("Current Models")
        choose("Trained with Different Person Validation")
        choose("ResNet-50 + LSTM + Attention")
        assertEquals(EmotionModelCatalog.legacy, EmotionModelCatalog.selected(preferences))

        // Recreates the activity and its dialog, as a screen rotation does.
        requireNotNull(scenario).recreate()

        onView(withText("ResNet-50 + LSTM + Attention — Trace Type"))
            .inRoot(isDialog())
            .check(matches(isDisplayed()))
        choose("Kalman Filter + CV")

        assertSelection(model("current_person_lstm_attention_kalman_cv"))
        assertEquals(Settings.Trace.Type.SPLINE_CV, Settings.Trace.inferenceType)
    }

    @Test
    fun selectingPreviousModelRestoresEditableInferenceTrace() {
        openPicker()
        choose("Current Models")
        choose("Trained with Different Person Validation")
        choose("ResNet-50 + LSTM + Attention")
        choose("Kalman Filter + CV")
        assertSelection(model("current_person_lstm_attention_kalman_cv"))

        openPicker()
        choose("Previous Models")
        choose("ResNet-50 + LSTM + Attention Model")

        assertSelection(EmotionModelCatalog.legacy)
        assertEquals("SPLINE_CV", preferences.getString(Settings.KEY_INFERENCE_TRACE_TYPE, null))
    }

    @Test
    fun backAndCancelLeaveTheModelAndTraceUnchanged() {
        val before = preferences.all.toMap()
        openPicker()
        choose("Current Models")
        choose("Trained with Different Person Validation")
        choose("ResNet-50 + LSTM + Attention")
        assertEquals(before, preferences.all)

        onView(withId(android.R.id.button3)).inRoot(isDialog()).perform(click())
        onView(withText("Trained with Different Person Validation"))
            .inRoot(isDialog())
            .check(matches(isDisplayed()))
        choose("Standalone ResNet-50")
        onView(withId(android.R.id.button2)).inRoot(isDialog()).perform(click())

        assertEquals(before, preferences.all)
        assertSelection(EmotionModelCatalog.legacy)
        assertEquals(Settings.Trace.Type.RAW, Settings.Trace.inferenceType)
    }

    private fun openPicker() {
        onView(withText(R.string.change_emotion_detection_model)).perform(click())
    }

    private fun choose(label: String) {
        onView(withText(label)).inRoot(isDialog()).perform(click())
    }

    private fun model(id: String): EmotionModelSpec = requireNotNull(EmotionModelCatalog.findById(id))

    private fun assertSelection(expected: EmotionModelSpec) {
        assertEquals(expected, EmotionModelCatalog.selected(preferences))
        onView(withText(expected.selectionSummary)).check(matches(isDisplayed()))
        requireNotNull(scenario).onActivity { activity ->
            val fragment = activity.supportFragmentManager
                .findFragmentById(R.id.settings_container) as SettingsActivity.SettingsFragment
            assertEquals(EmotionModelCatalog.KEY_SELECTION, fragment.preferenceScreen.getPreference(0).key)
            assertEquals(
                expected.selectionSummary,
                fragment.findPreference<Preference>(EmotionModelCatalog.KEY_SELECTION)?.summary.toString()
            )
            val tracePreference = requireNotNull(
                fragment.findPreference<ListPreference>(Settings.KEY_INFERENCE_TRACE_TYPE)
            )
            assertEquals(expected.isLegacy, tracePreference.isEnabled)
            expected.requiredTraceType?.let { traceType ->
                assertEquals(traceType.name, preferences.getString(Settings.KEY_INFERENCE_TRACE_TYPE, null))
                assertEquals(traceType.name, tracePreference.value)
                assertTrue(tracePreference.summary.toString().contains("required by the selected model"))
            }
        }
    }
}
