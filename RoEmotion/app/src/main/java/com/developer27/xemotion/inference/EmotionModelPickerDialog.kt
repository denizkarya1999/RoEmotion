package com.developer27.xemotion.inference

import android.app.Dialog
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.DialogFragment
import androidx.preference.PreferenceManager
import com.developer27.xemotion.R

/** A single restorable dialog keeps unfinished choices out of saved app settings. */
class EmotionModelPickerDialog : DialogFragment() {
    private enum class Step { ROOT, PREVIOUS, DISTRIBUTION, ARCHITECTURE, TRACE }

    private var step = Step.ROOT
    private var distribution = EmotionTrainingDistribution.RANDOM
    private var architecture = EmotionModelArchitecture.STANDALONE

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        savedInstanceState?.let { state ->
            step = Step.valueOf(state.getString(STATE_STEP) ?: Step.ROOT.name)
            distribution = EmotionTrainingDistribution.valueOf(
                state.getString(STATE_DISTRIBUTION) ?: EmotionTrainingDistribution.RANDOM.name
            )
            architecture = EmotionModelArchitecture.valueOf(
                state.getString(STATE_ARCHITECTURE) ?: EmotionModelArchitecture.STANDALONE.name
            )
        }
    }

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
        AlertDialog.Builder(requireContext())
            .setTitle(title())
            .setSingleChoiceItems(choices().toTypedArray(), checkedChoice()) { _, index ->
                choose(index)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .setNeutralButton(R.string.emotion_model_back, null)
            .create()

    override fun onStart() {
        super.onStart()
        val alert = requireDialog() as AlertDialog
        alert.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener { goBack() }
        renderStep()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString(STATE_STEP, step.name)
        outState.putString(STATE_DISTRIBUTION, distribution.name)
        outState.putString(STATE_ARCHITECTURE, architecture.name)
        super.onSaveInstanceState(outState)
    }

    private fun title(): String = when (step) {
        Step.ROOT -> getString(R.string.change_emotion_detection_model)
        Step.PREVIOUS -> getString(R.string.emotion_previous_models)
        Step.DISTRIBUTION -> getString(R.string.emotion_current_models)
        Step.ARCHITECTURE -> distribution.displayName
        Step.TRACE -> getString(R.string.emotion_model_trace_title, architecture.displayName)
    }

    private fun choices(): List<String> = when (step) {
        Step.ROOT -> listOf(
            getString(R.string.emotion_previous_models),
            getString(R.string.emotion_current_models)
        )
        Step.PREVIOUS -> listOf(EmotionModelCatalog.legacy.displayName)
        Step.DISTRIBUTION -> EmotionTrainingDistribution.entries.map { it.displayName }
        Step.ARCHITECTURE -> EmotionModelArchitecture.entries.map { it.displayName }
        Step.TRACE -> EmotionTraceVariant.entries.map { it.displayName }
    }

    private fun checkedChoice(): Int {
        val selected = EmotionModelCatalog.selected(preferences())
        return when (step) {
            Step.ROOT -> if (selected.isLegacy) 0 else 1
            Step.PREVIOUS -> if (selected.isLegacy) 0 else -1
            Step.DISTRIBUTION -> selected.trainingDistribution?.ordinal ?: -1
            Step.ARCHITECTURE -> if (selected.trainingDistribution == distribution) {
                selected.architecture.ordinal
            } else -1
            Step.TRACE -> if (selected.trainingDistribution == distribution &&
                selected.architecture == architecture
            ) selected.traceVariant?.ordinal ?: -1 else -1
        }
    }

    private fun choose(index: Int) {
        when (step) {
            Step.ROOT -> step = if (index == 0) Step.PREVIOUS else Step.DISTRIBUTION
            Step.PREVIOUS -> {
                finishSelection(EmotionModelCatalog.legacy)
                return
            }
            Step.DISTRIBUTION -> {
                distribution = EmotionTrainingDistribution.entries[index]
                step = Step.ARCHITECTURE
            }
            Step.ARCHITECTURE -> {
                architecture = EmotionModelArchitecture.entries[index]
                step = Step.TRACE
            }
            Step.TRACE -> {
                val trace = EmotionTraceVariant.entries[index]
                finishSelection(EmotionModelCatalog.models.first {
                    it.trainingDistribution == distribution && it.architecture == architecture &&
                        it.traceVariant == trace
                })
                return
            }
        }
        renderStep()
    }

    private fun goBack() {
        step = when (step) {
            Step.ROOT -> return
            Step.PREVIOUS, Step.DISTRIBUTION -> Step.ROOT
            Step.ARCHITECTURE -> Step.DISTRIBUTION
            Step.TRACE -> Step.ARCHITECTURE
        }
        renderStep()
    }

    private fun renderStep() {
        val alert = dialog as? AlertDialog ?: return
        alert.setTitle(title())
        alert.listView.apply {
            adapter = ArrayAdapter(alert.context, android.R.layout.simple_list_item_single_choice, choices())
            choiceMode = ListView.CHOICE_MODE_SINGLE
            clearChoices()
            val checked = checkedChoice()
            if (checked >= 0) setItemChecked(checked, true)
            setOnItemClickListener { _, _, position, _ -> choose(position) }
        }
        alert.getButton(AlertDialog.BUTTON_NEUTRAL)?.visibility =
            if (step == Step.ROOT) View.GONE else View.VISIBLE
    }

    private fun finishSelection(model: EmotionModelSpec) {
        EmotionModelCatalog.select(preferences(), model)
        parentFragmentManager.setFragmentResult(RESULT_KEY, Bundle())
        Toast.makeText(
            requireContext(),
            getString(R.string.emotion_model_selected, model.displayName),
            Toast.LENGTH_SHORT
        ).show()
        dismiss()
    }

    private fun preferences() = PreferenceManager.getDefaultSharedPreferences(requireContext())

    companion object {
        const val TAG = "emotion_model_picker"
        const val RESULT_KEY = "emotion_model_selection_changed"
        private const val STATE_STEP = "step"
        private const val STATE_DISTRIBUTION = "distribution"
        private const val STATE_ARCHITECTURE = "architecture"
    }
}
