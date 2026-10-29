package com.developer27.xemotion.ar

import android.os.Bundle
import android.os.Build
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import com.developer27.xemotion.camera.CameraHelper
import com.developer27.xemotion.databinding.ActivityMainBinding

/** Instrumentation host: deliberately avoids model loading and opening the camera. */
class CardboardTestActivity : AppCompatActivity() {
    lateinit var binding: ActivityMainBinding
    lateinit var cameraHelper: CameraHelper
    lateinit var manager: ArModeManager
    var processing = false
    var allowStart = true
    var starts = 0
    var stops = 0
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Show only this test window above the lock screen; never dismiss the keyguard.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        cameraHelper = CameraHelper(this, binding, getSharedPreferences("cardboard_test_camera", MODE_PRIVATE))
        manager = ArModeManager(this, binding, cameraHelper,
            { processing }, { starts++; processing = allowStart }, { stops++; processing = false })
    }
    override fun onPause() {
        manager.onPause()
        super.onPause()
    }
}
