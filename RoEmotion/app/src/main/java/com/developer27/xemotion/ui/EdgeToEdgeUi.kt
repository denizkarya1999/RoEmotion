package com.developer27.xemotion.ui

import android.graphics.Color
import android.view.View
import android.view.ViewGroup
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

/** Shared edge-to-edge setup for Android 15+ and matching behavior on older releases. */
fun ComponentActivity.enableRoEmotionEdgeToEdge() {
    enableEdgeToEdge(
        statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
        navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT)
    )
}

/** Adds system-bar and cutout insets without discarding padding declared in XML. */
fun View.applySystemBarPadding(
    start: Boolean = true,
    top: Boolean = true,
    end: Boolean = true,
    bottom: Boolean = true
) {
    val initialStart = paddingStart
    val initialTop = paddingTop
    val initialEnd = paddingEnd
    val initialBottom = paddingBottom
    ViewCompat.setOnApplyWindowInsetsListener(this) { view, windowInsets ->
        val insets = windowInsets.getInsets(
            WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
        )
        val isRtl = view.layoutDirection == View.LAYOUT_DIRECTION_RTL
        val startInset = if (isRtl) insets.right else insets.left
        val endInset = if (isRtl) insets.left else insets.right
        view.setPaddingRelative(
            initialStart + if (start) startInset else 0,
            initialTop + if (top) insets.top else 0,
            initialEnd + if (end) endInset else 0,
            initialBottom + if (bottom) insets.bottom else 0
        )
        windowInsets
    }
    ViewCompat.requestApplyInsets(this)
}

/** Adds selected system-bar and cutout insets to XML margins. */
fun View.applySystemBarMargins(
    start: Boolean = false,
    top: Boolean = false,
    end: Boolean = false,
    bottom: Boolean = false
) {
    val params = layoutParams as? ViewGroup.MarginLayoutParams ?: return
    val initialStart = params.marginStart
    val initialTop = params.topMargin
    val initialEnd = params.marginEnd
    val initialBottom = params.bottomMargin
    ViewCompat.setOnApplyWindowInsetsListener(this) { view, windowInsets ->
        val insets = windowInsets.getInsets(
            WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
        )
        val isRtl = view.layoutDirection == View.LAYOUT_DIRECTION_RTL
        val startInset = if (isRtl) insets.right else insets.left
        val endInset = if (isRtl) insets.left else insets.right
        (view.layoutParams as? ViewGroup.MarginLayoutParams)?.let { margins ->
            margins.marginStart = initialStart + if (start) startInset else 0
            margins.topMargin = initialTop + if (top) insets.top else 0
            margins.marginEnd = initialEnd + if (end) endInset else 0
            margins.bottomMargin = initialBottom + if (bottom) insets.bottom else 0
            view.layoutParams = margins
        }
        windowInsets
    }
    ViewCompat.requestApplyInsets(this)
}
