package com.developer27.xemotion.inference

/** Keeps chronological endpoints; only the previous model requires synthetic repeated frames. */
internal fun <T> sampleEmotionSequence(
    items: List<T>,
    maximumFrames: Int,
    padShortSequence: Boolean
): List<T> {
    require(items.isNotEmpty()) { "At least one image is required." }
    require(maximumFrames > 0) { "The frame limit must be positive." }
    return when {
        maximumFrames == 1 -> listOf(items.first())
        items.size > maximumFrames -> List(maximumFrames) { index ->
            items[(index * items.lastIndex.toDouble() / (maximumFrames - 1)).toInt()]
        }
        items.size < maximumFrames && padShortSequence ->
            items + List(maximumFrames - items.size) { items.last() }
        else -> items
    }
}
