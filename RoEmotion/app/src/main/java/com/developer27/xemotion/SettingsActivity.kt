package com.developer27.xemotion

import android.content.Intent
import android.content.pm.ActivityInfo
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.EditTextPreference
import androidx.preference.ListPreference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SwitchPreference
import com.developer27.xemotion.videoprocessing.Settings

class SettingsActivity : AppCompatActivity() {

    // Initialize activity and load settings UI
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Disable screen rotation (lock to portrait)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        // Prevent screen hibernation (keep screen on)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Use the updated layout with a fixed header.
        setContentView(R.layout.settings_activity)

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

            // Rolling Shutter Speed listener.
            val shutterSpeedPref = findPreference<ListPreference>("shutter_speed")
            shutterSpeedPref?.setOnPreferenceChangeListener { _, newValue ->
                // For example, update your global shutter speed setting here.
                Toast.makeText(
                    context,
                    "Rolling Shutter Speed set to ${newValue} Hz",
                    Toast.LENGTH_SHORT
                ).show()
                true
            }

            // ISO numeric input preference.
            val isoPref = findPreference<EditTextPreference>("iso_value")
            isoPref?.setOnBindEditTextListener { edit ->
                // numeric only from XML; optionally add min/max hints
                edit.hint = "100–6400"
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

            // Detection mode listener.
            val detectionModePref = findPreference<ListPreference>("detection_mode")
            detectionModePref?.setOnPreferenceChangeListener { _, newValue ->
                when (newValue as String) {
                    "CONTOUR" -> {
                        Settings.DetectionMode.current = Settings.DetectionMode.Mode.CONTOUR
                        Settings.DetectionMode.enableYOLOinference = false
                    }

                    "YOLO" -> {
                        Settings.DetectionMode.current = Settings.DetectionMode.Mode.YOLO
                        Settings.DetectionMode.enableYOLOinference = true
                    }

                    else -> {
                        Settings.DetectionMode.current = Settings.DetectionMode.Mode.CONTOUR
                        Settings.DetectionMode.enableYOLOinference = false
                    }
                }
                Toast.makeText(context, "Detection mode set to $newValue", Toast.LENGTH_SHORT)
                    .show()
                true
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

            // RAW trace enable listener.
            val rawTracePref = findPreference<SwitchPreference>("enable_raw_trace")
            rawTracePref?.setOnPreferenceChangeListener { _, newValue ->
                val enabled = newValue as Boolean
                Settings.Trace.enableRAWtrace = enabled
                Toast.makeText(
                    context,
                    "RAW Trace: ${if (enabled) "Yes" else "No"}",
                    Toast.LENGTH_SHORT
                ).show()
                true
            }

            // SPLINE trace enable listener.
            val splineTracePref = findPreference<SwitchPreference>("enable_spline_trace")
            splineTracePref?.setOnPreferenceChangeListener { _, newValue ->
                val enabled = newValue as Boolean
                Settings.Trace.enableSPLINEtrace = enabled
                Toast.makeText(
                    context,
                    "SPLINE Trace: ${if (enabled) "Yes" else "No"}",
                    Toast.LENGTH_SHORT
                ).show()
                true
            }

            // Export Data: 28x28 IMG saving listener.
            val frameImgPref = findPreference<SwitchPreference>("frame_img")
            frameImgPref?.setOnPreferenceChangeListener { _, newValue ->
                val enabled = newValue as Boolean
                Settings.ExportData.frameIMG = enabled
                Toast.makeText(
                    context,
                    "28x28 IMG Saving: ${if (enabled) "Yes" else "No"}",
                    Toast.LENGTH_SHORT
                ).show()
                true
            }

            // Export Data: Video saving listener.
            val videoDataPref = findPreference<SwitchPreference>("video_data")
            videoDataPref?.setOnPreferenceChangeListener { _, newValue ->
                val enabled = newValue as Boolean
                Settings.ExportData.enablePredictionLogging = enabled
                Toast.makeText(
                    context,
                    "Video Saving: ${if (enabled) "Yes" else "No"}",
                    Toast.LENGTH_SHORT
                ).show()
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
        }
    }

    // Handle back press: return to previous activity and apply settings
    override fun onBackPressed() {
        super.onBackPressed()
        // Save settings and return to the calling Activity.
        setResult(RESULT_OK, Intent())
        finish()
    }
}