package com.developer27.xemotion

import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.EditTextPreference
import androidx.preference.ListPreference
import androidx.preference.MultiSelectListPreference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SeekBarPreference
import androidx.preference.SwitchPreference
import com.developer27.xemotion.ui.applySystemBarPadding
import com.developer27.xemotion.ui.enableRoEmotionEdgeToEdge
import com.developer27.xemotion.videoprocessing.Settings

class SettingsActivity : AppCompatActivity() {

    // Initialize activity and load settings UI
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableRoEmotionEdgeToEdge()

        // Prevent screen hibernation (keep screen on)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Use the updated layout with a fixed header.
        setContentView(R.layout.settings_activity)
        findViewById<android.view.View>(R.id.settings_root).applySystemBarPadding()

        // Attach SettingsFragment to display preference UI inside the container
        supportFragmentManager
            .beginTransaction()
            .replace(R.id.settings_container, SettingsFragment())
            .commit()
    }

    // Fragment that manages all preference settings
    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            // Load preferences from the XML resource.
            setPreferencesFromResource(R.xml.root_preferences, rootKey)
            preferenceManager.sharedPreferences?.let(Settings::load)

            val operatingModePref = findPreference<ListPreference>(Settings.KEY_OPERATING_MODE)
            operatingModePref?.setOnPreferenceChangeListener { _, newValue ->
                val mode = runCatching {
                    Settings.OperatingMode.Mode.valueOf(newValue as String)
                }.getOrDefault(Settings.OperatingMode.Mode.INFERENCE)
                Settings.applyOperatingMode(mode)
                updateModeSpecificPreferences(mode)
                Toast.makeText(
                    context,
                    if (mode == Settings.OperatingMode.Mode.DATA_COLLECTION) {
                        "Data Collection Mode: contour tracking and automatic trace export"
                    } else {
                        "Inference Mode: YOLO, ResNet-50, and AR"
                    },
                    Toast.LENGTH_LONG
                ).show()
                true
            }

            // Rolling Shutter Speed listener.
            val shutterSpeedPref = findPreference<ListPreference>("shutter_speed")
            shutterSpeedPref?.setOnPreferenceChangeListener { _, newValue ->
                Settings.RollingShutter.speedHz = newValue.toString().toFloatOrNull() ?: 60f
                // For example, update your global shutter speed setting here.
                Toast.makeText(
                    context,
                    "Rolling Shutter Speed set to ${newValue} Hz",
                    Toast.LENGTH_SHORT
                ).show()
                true
            }

            // ISO numeric input preference.
            val manualIsoPref = findPreference<SwitchPreference>("manual_iso_enabled")
            val isoPref = findPreference<EditTextPreference>("iso_value")
            isoPref?.setOnBindEditTextListener { edit ->
                // numeric only from XML; optionally add min/max hints
                edit.hint = "100–6400"
            }
            isoPref?.isEnabled = manualIsoPref?.isChecked == true
            manualIsoPref?.setOnPreferenceChangeListener { _, newValue ->
                isoPref?.isEnabled = newValue as? Boolean == true
                true
            }

            // Parse ISO input, enforce safe range, and store value
            isoPref?.setOnPreferenceChangeListener { _, newValue ->
                val entered = (newValue as? String)?.trim().orEmpty()
                val iso = entered.toIntOrNull()
                val clamped = when {
                    iso == null -> null
                    iso < 50 -> 50               // soft min (safer for low light)
                    iso > 25600 -> 25600         // soft max; actual max will be clamped by camera
                    else -> iso
                }
                if (clamped == null) {
                    Toast.makeText(context, "Please enter a valid ISO number.", Toast.LENGTH_SHORT).show()
                    false
                } else {
                    // Store the (soft) clamped value
                    isoPref.text = clamped.toString()
                    Toast.makeText(context, "ISO set to $clamped", Toast.LENGTH_SHORT).show()
                    true
                }
            }

            // Maximum bounding boxes preference.
            val maxBoxesPref = findPreference<ListPreference>("max_boxes")
            maxBoxesPref?.setOnPreferenceChangeListener { _, newValue ->
                val n = (newValue as? String)?.toIntOrNull()?.coerceIn(1, 3) ?: 3
                Settings.BoundingBox.maxPerFrame = n
                Toast.makeText(context, "Max bounding boxes: $n", Toast.LENGTH_SHORT).show()
                true
            }

            // Bounding box enable listener.
            val boundingBoxPref = findPreference<SwitchPreference>("enable_bounding_box")
            boundingBoxPref?.setOnPreferenceChangeListener { _, newValue ->
                val enabled = newValue as Boolean
                Settings.BoundingBox.enableBoundingBox = enabled
                Toast.makeText(
                    context,
                    "Bounding Box: ${if (enabled) "Yes" else "No"}",
                    Toast.LENGTH_SHORT
                ).show()
                true
            }

            val traceTypesPref = findPreference<MultiSelectListPreference>(Settings.KEY_TRACE_TYPES)
            traceTypesPref?.summary = traceTypeSummary(traceTypesPref.values)
            traceTypesPref?.setOnPreferenceChangeListener { preference, newValue ->
                @Suppress("UNCHECKED_CAST")
                val values = newValue as? Set<String> ?: emptySet()
                val types = values.mapNotNull { value ->
                    runCatching { Settings.Trace.Type.valueOf(value) }.getOrNull()
                }.toSet()
                if (types.isEmpty()) {
                    Toast.makeText(context, "Select at least one trace type.", Toast.LENGTH_SHORT).show()
                    false
                } else {
                    Settings.Trace.updateCollectionTypes(types)
                    preference.summary = traceTypeSummary(values)
                    true
                }
            }

            findPreference<ListPreference>(Settings.KEY_INFERENCE_TRACE_TYPE)
                ?.setOnPreferenceChangeListener { _, newValue ->
                    val type = (newValue as? String)
                        ?.let { value -> runCatching { Settings.Trace.Type.valueOf(value) }.getOrNull() }
                        ?: Settings.Trace.Type.SPLINE_CV
                    Settings.Trace.updateInferenceType(type)
                    true
                }

            findPreference<SeekBarPreference>(Settings.KEY_TRACE_BOLDNESS)
                ?.setOnPreferenceChangeListener { _, newValue ->
                    Settings.Trace.updateBoldness(newValue as? Int ?: Settings.Trace.DEFAULT_BOLDNESS)
                    true
                }

            findPreference<SeekBarPreference>(Settings.KEY_INFERENCE_LABEL_SIZE)
                ?.setOnPreferenceChangeListener { _, newValue ->
                    Settings.Inference.updateLabelSize(
                        newValue as? Int ?: Settings.Inference.DEFAULT_LABEL_SIZE
                    )
                    true
                }

            // Prediction Logging listener
            val predLoggingPref = findPreference<SwitchPreference>("enable_prediction_logging")
            predLoggingPref?.setOnPreferenceChangeListener { _, newValue ->
                val enabled = newValue as Boolean
                Settings.ExportData.enablePredictionLogging = enabled
                Toast.makeText(
                    context,
                    "Prediction Logging: ${if (enabled) "Enabled" else "Disabled"}",
                    Toast.LENGTH_SHORT
                ).show()
                true
            }

            updateModeSpecificPreferences(Settings.OperatingMode.current)
        }

        private fun traceTypeSummary(values: Set<String>): String {
            val names = Settings.Trace.Type.entries
                .filter { it.name in values }
                .map(Settings.Trace.Type::displayName)
            return names.joinToString().ifBlank { "Select at least one trace type" }
        }

        private fun updateModeSpecificPreferences(mode: Settings.OperatingMode.Mode) {
            val capabilities = Settings.capabilitiesFor(mode)
            findPreference<MultiSelectListPreference>(Settings.KEY_TRACE_TYPES)?.isEnabled =
                capabilities.traceCollectionSettingsEnabled
            findPreference<ListPreference>(Settings.KEY_INFERENCE_TRACE_TYPE)?.isEnabled =
                capabilities.inferenceTraceSettingsEnabled
            findPreference<SeekBarPreference>(Settings.KEY_INFERENCE_LABEL_SIZE)?.isEnabled =
                capabilities.inferenceSettingsEnabled
            findPreference<ListPreference>("max_boxes")?.isEnabled =
                capabilities.inferenceSettingsEnabled
            findPreference<SwitchPreference>("enable_prediction_logging")?.isEnabled =
                capabilities.inferenceSettingsEnabled
        }
    }

}
