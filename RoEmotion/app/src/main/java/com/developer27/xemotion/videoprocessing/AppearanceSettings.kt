package com.developer27.xemotion.videoprocessing

import android.content.SharedPreferences

/** Independent, persisted live-display profiles. Export/model-input settings are separate. */
object AppearanceSettings {
    enum class Mode(val prefix: String) { CV("cv"), AR("ar") }
    enum class Field(val suffix: String, val minimum: Int, val maximum: Int) {
        BOX("box_thickness", 1, 20), TEXT("text_size", 4, 20), TRACE("trace_thickness", 1, 20)
    }
    data class Profile(val boxThickness: Int, val textSize: Int, val traceThickness: Int) {
        fun value(field: Field): Int = when (field) {
            Field.BOX -> boxThickness
            Field.TEXT -> textSize
            Field.TRACE -> traceThickness
        }
        fun updated(field: Field, value: Int): Profile {
            val safe = value.coerceIn(field.minimum, field.maximum)
            return when (field) {
                Field.BOX -> copy(boxThickness = safe)
                Field.TEXT -> copy(textSize = safe)
                Field.TRACE -> copy(traceThickness = safe)
            }
        }
    }
    val DEFAULT_CV = Profile(10, 16, 5)
    val DEFAULT_AR = Profile(3, 8, 2)
    @Volatile private var cv = DEFAULT_CV
    @Volatile private var ar = DEFAULT_AR

    fun key(mode: Mode, field: Field): String = "${mode.prefix}_${field.suffix}"
    fun profile(mode: Mode): Profile = if (mode == Mode.AR) ar else cv

    @Synchronized
    fun update(mode: Mode, field: Field, value: Int) {
        val next = profile(mode).updated(field, value)
        if (mode == Mode.AR) ar = next else cv = next
    }

    /** Run before XML preference inflation, which otherwise seeds fresh defaults first. */
    fun migrateLegacy(preferences: SharedPreferences) {
        val legacyCv = DEFAULT_CV
            .updated(Field.TEXT, readInt(preferences, Settings.KEY_INFERENCE_LABEL_SIZE, DEFAULT_CV.textSize))
            .updated(Field.TRACE, readInt(preferences, Settings.KEY_TRACE_BOLDNESS, DEFAULT_CV.traceThickness))
        val editor = preferences.edit()
        var changed = false
        for (mode in Mode.entries) for (field in Field.entries) {
            val key = key(mode, field)
            if (!preferences.contains(key)) {
                val defaults = if (mode == Mode.AR) DEFAULT_AR else legacyCv
                editor.putInt(key, defaults.value(field))
                changed = true
            }
        }
        if (changed) editor.apply()
    }

    fun load(preferences: SharedPreferences) {
        migrateLegacy(preferences)
        for (mode in Mode.entries) for (field in Field.entries) {
            val defaults = if (mode == Mode.AR) DEFAULT_AR else DEFAULT_CV
            update(mode, field, readInt(preferences, key(mode, field), defaults.value(field)))
        }
    }

    private fun readInt(preferences: SharedPreferences, key: String, fallback: Int): Int =
        runCatching { preferences.getInt(key, fallback) }.getOrDefault(fallback)
}
