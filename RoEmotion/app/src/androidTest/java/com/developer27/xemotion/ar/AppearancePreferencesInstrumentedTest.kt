package com.developer27.xemotion.ar

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.developer27.xemotion.videoprocessing.AppearanceSettings
import com.developer27.xemotion.videoprocessing.AppearanceSettings.Field
import com.developer27.xemotion.videoprocessing.AppearanceSettings.Mode
import com.developer27.xemotion.videoprocessing.Settings
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AppearancePreferencesInstrumentedTest {
    private val preferences = InstrumentationRegistry.getInstrumentation().targetContext
        .getSharedPreferences("appearance-regression-tests", Context.MODE_PRIVATE)
    private lateinit var originals: Map<Mode, AppearanceSettings.Profile>

    @Before fun setup() {
        originals = Mode.entries.associateWith(AppearanceSettings::profile)
        preferences.edit().clear().commit()
    }
    @After fun cleanup() {
        preferences.edit().clear().commit()
        originals.forEach { (mode, profile) ->
            Field.entries.forEach { AppearanceSettings.update(mode, it, profile.value(it)) }
        }
    }

    @Test fun migrationPreservesLegacyCvAndExistingArChoicesWithoutRecopying() {
        preferences.edit().putInt(Settings.KEY_INFERENCE_LABEL_SIZE, 20)
            .putInt(Settings.KEY_TRACE_BOLDNESS, 9)
            .putInt(AppearanceSettings.key(Mode.AR, Field.TEXT), 6).commit()
        AppearanceSettings.load(preferences)
        assertEquals(AppearanceSettings.Profile(10, 20, 9), AppearanceSettings.profile(Mode.CV))
        assertEquals(AppearanceSettings.Profile(3, 6, 2), AppearanceSettings.profile(Mode.AR))
        preferences.edit().putInt(Settings.KEY_INFERENCE_LABEL_SIZE, 8)
            .putInt(Settings.KEY_TRACE_BOLDNESS, 2).commit()
        AppearanceSettings.load(preferences)
        assertEquals(AppearanceSettings.Profile(10, 20, 9), AppearanceSettings.profile(Mode.CV))
        assertEquals(AppearanceSettings.Profile(3, 6, 2), AppearanceSettings.profile(Mode.AR))
    }

    @Test fun savedProfilesReloadIndependentlyAndDoNotChangeExports() {
        val cv = AppearanceSettings.Profile(7, 14, 9)
        val ar = AppearanceSettings.Profile(2, 5, 1)
        val export = Settings.Trace.exportLineThickness
        val editor = preferences.edit()
        for (mode in Mode.entries) for (field in Field.entries) {
            editor.putInt(AppearanceSettings.key(mode, field), (if (mode == Mode.AR) ar else cv).value(field))
        }
        assertTrue(editor.commit())
        AppearanceSettings.load(preferences)
        assertEquals(cv, AppearanceSettings.profile(Mode.CV))
        assertEquals(ar, AppearanceSettings.profile(Mode.AR))
        preferences.edit().putInt(AppearanceSettings.key(Mode.AR, Field.BOX), 4).commit()
        AppearanceSettings.load(preferences)
        assertEquals(cv, AppearanceSettings.profile(Mode.CV))
        assertEquals(ar.copy(boxThickness = 4), AppearanceSettings.profile(Mode.AR))
        assertEquals(export, Settings.Trace.exportLineThickness)
    }
}
