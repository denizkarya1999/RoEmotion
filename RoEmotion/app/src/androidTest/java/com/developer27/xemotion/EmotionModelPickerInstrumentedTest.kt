package com.developer27.xemotion

import android.app.Activity
import android.app.Application
import android.content.SharedPreferences
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.content.ContextCompat
import androidx.fragment.app.DialogFragment
import androidx.lifecycle.Lifecycle
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
import java.io.File

/** Exercises the real Options screen without initializing camera or model runtimes. */
@RunWith(AndroidJUnit4::class)
class EmotionModelPickerInstrumentedTest {
    private lateinit var preferences: SharedPreferences
    private var originalPreferences = emptyMap<String, Any?>()
    private var scenario: ActivityScenario<SettingsActivity>? = null
    private var originalNightMode = AppCompatDelegate.MODE_NIGHT_UNSPECIFIED
    private var testApplication: Application? = null
    private val visibleTestActivity = object : Application.ActivityLifecycleCallbacks {
        override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {
            if (activity !is SettingsActivity) return
            // Keep this test's own window visible without dismissing the device's lock screen.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                activity.setShowWhenLocked(true)
                activity.setTurnScreenOn(true)
            } else {
                @Suppress("DEPRECATION")
                activity.window.addFlags(
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                        WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
                )
            }
            activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }

        override fun onActivityStarted(activity: Activity) = Unit
        override fun onActivityResumed(activity: Activity) = Unit
        override fun onActivityPaused(activity: Activity) = Unit
        override fun onActivityStopped(activity: Activity) = Unit
        override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit
        override fun onActivityDestroyed(activity: Activity) = Unit
    }

    @Before
    fun openOptionsWithOriginalModel() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        testApplication = context.applicationContext as Application
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            testApplication?.registerActivityLifecycleCallbacks(visibleTestActivity)
        }
        originalNightMode = AppCompatDelegate.getDefaultNightMode()
        preferences = PreferenceManager.getDefaultSharedPreferences(context)
        originalPreferences = preferences.all.mapValues { (_, value) ->
            if (value is Set<*>) value.toSet() else value
        }
        assertTrue(
            preferences.edit().clear()
                .putString(Settings.KEY_OPERATING_MODE, Settings.OperatingMode.Mode.INFERENCE.name)
                .commit()
        )
        launchOptions()
    }

    @After
    fun restoreOriginalPreferences() {
        try {
            scenario?.close()
        } finally {
            InstrumentationRegistry.getInstrumentation().runOnMainSync {
                testApplication?.unregisterActivityLifecycleCallbacks(visibleTestActivity)
            }
            testApplication = null
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
                    AppCompatDelegate.setDefaultNightMode(originalNightMode)
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

    @Test
    fun darkDialogsUseReadableColorsAndDayDialogsKeepTheirAppearance() {
        relaunchWithNightMode(AppCompatDelegate.MODE_NIGHT_YES)
        openPicker()
        captureDialog("dialog-dark.png", dark = true)
        onView(withId(android.R.id.button2)).inRoot(isDialog()).perform(click())

        // App Mode uses AndroidX's standard preference dialog, not the model picker.
        requireNotNull(scenario).onActivity { activity ->
            val fragment = activity.supportFragmentManager
                .findFragmentById(R.id.settings_container) as SettingsActivity.SettingsFragment
            fragment.onDisplayPreferenceDialog(
                requireNotNull(fragment.findPreference<ListPreference>(Settings.KEY_OPERATING_MODE))
            )
        }
        onView(withText("App Mode")).inRoot(isDialog()).check(matches(isDisplayed()))
        captureDialog("preference-dialog-dark.png", dark = true)
        onView(withId(android.R.id.button2)).inRoot(isDialog()).perform(click())

        // Numeric preferences inflate their EditText using the activity theme.
        requireNotNull(scenario).onActivity { activity ->
            val fragment = activity.supportFragmentManager
                .findFragmentById(R.id.settings_container) as SettingsActivity.SettingsFragment
            fragment.onDisplayPreferenceDialog(
                requireNotNull(fragment.findPreference<Preference>("iso_value"))
            )
        }
        onView(withText("Set ISO")).inRoot(isDialog()).check(matches(isDisplayed()))
        requireNotNull(scenario).onActivity { activity ->
            val dialog = activeDialog(activity)
            val editor = requireNotNull(dialog.findViewById<TextView>(android.R.id.edit))
            assertEquals(ContextCompat.getColor(activity, R.color.dialog_night_text), editor.currentTextColor)
        }
        onView(withId(android.R.id.button2)).inRoot(isDialog()).perform(click())

        relaunchWithNightMode(AppCompatDelegate.MODE_NIGHT_NO)
        openPicker()
        captureDialog("dialog-light.png", dark = false)
    }

    private fun relaunchWithNightMode(mode: Int) {
        scenario?.close()
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            AppCompatDelegate.setDefaultNightMode(mode)
        }
        launchOptions()
    }

    private fun launchOptions() {
        scenario = ActivityScenario.launch(SettingsActivity::class.java)
            .moveToState(Lifecycle.State.RESUMED)
        InstrumentationRegistry.getInstrumentation().waitForIdleSync()
        onView(withText(R.string.change_emotion_detection_model)).check(matches(isDisplayed()))
    }

    private fun activeDialog(activity: SettingsActivity): AlertDialog =
        requireNotNull(
            activity.supportFragmentManager.fragments.filterIsInstance<DialogFragment>()
                .lastOrNull { it.dialog?.isShowing == true }?.dialog as? AlertDialog
        )

    private fun captureDialog(name: String, dark: Boolean) {
        requireNotNull(scenario).onActivity { activity ->
            val actualNight = activity.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK
            assertEquals(if (dark) Configuration.UI_MODE_NIGHT_YES else Configuration.UI_MODE_NIGHT_NO, actualNight)
            val dialog = activeDialog(activity)
            val decor = requireNotNull(dialog.window).decorView
            assertTrue("Dialog has not been laid out", decor.width > 0 && decor.height > 0)
            val bitmap = Bitmap.createBitmap(decor.width, decor.height, Bitmap.Config.ARGB_8888)
            try {
                decor.draw(Canvas(bitmap))
                val directory = requireNotNull(activity.getExternalFilesDir("qa"))
                assertTrue(directory.isDirectory || directory.mkdirs())
                File(directory, name).outputStream().use { output ->
                    assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
                }
                if (dark) {
                    val textColor = ContextCompat.getColor(activity, R.color.dialog_night_text)
                    val background = ContextCompat.getColor(activity, R.color.dialog_night_background)
                    assertEquals(textColor, requireNotNull(dialog.findViewById<TextView>(androidx.appcompat.R.id.alertTitle)).currentTextColor)
                    assertEquals(textColor, (dialog.listView.getChildAt(0) as TextView).currentTextColor)
                    assertEquals(textColor, dialog.getButton(AlertDialog.BUTTON_NEGATIVE).currentTextColor)
                    var darkSamples = 0
                    var samples = 0
                    for (x in 0 until bitmap.width step 8) {
                        for (y in 0 until bitmap.height step 8) {
                            samples++
                            if (bitmap.getPixel(x, y) == background) darkSamples++
                        }
                    }
                    assertTrue("Dialog surface must use the dark background", darkSamples > samples / 3)
                }
            } finally {
                bitmap.recycle()
            }
        }
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
