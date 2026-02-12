package com.example.lslclassifier

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.Executor

class SignAnalyzer(
    private val inferenceExecutor: Executor,
    private val classifier: OrtClassifier,
    // labelText = what you want to display, score = best score/logit
    private val onPrediction: (labelText: String, bestScore: Float) -> Unit,
) : ImageAnalysis.Analyzer {

    // ✅ Your 10 class labels (correct order)
    private val labels = listOf(
        "وداعا",
        "صباح الخير",
        "كيفك",
        "الحمدلله",
        "أنا",
        "لا",
        "أرجوك",
        "أسف",
        "شكراً",
        "نعم"
    )

    private var lastFrame: Bitmap? = null
    private var lastTimeMs: Long = 0L

    @Volatile private var busy = false

    override fun analyze(image: ImageProxy) {
        try {
            val now = System.currentTimeMillis()

            // Convert to Bitmap + rotate to upright
            val bmp = image.toBitmap().rotate(image.imageInfo.rotationDegrees.toFloat())

            if (lastFrame == null) {
                lastFrame = bmp
                lastTimeMs = now
                return
            }

            // Wait until we have ~1 second gap
            if (now - lastTimeMs < 1000L) return

            // Don’t queue multiple inferences
            if (busy) return
            busy = true

            val frameA = lastFrame!!   // older
            val frameB = bmp           // newer

            // Update stored frame for next pair
            lastFrame = bmp
            lastTimeMs = now

            inferenceExecutor.execute {
                try {
                    val inputA = Preprocessor.preprocess(frameA, 256)
                    val inputB = Preprocessor.preprocess(frameB, 256)

                    val logits = classifier.runDual(inputA, inputB)

                    // ✅ Debug: confirm output size is 10
                    Log.d("ORT", "output size=${logits.size} values=${logits.joinToString(limit = 10)}")

                    // ✅ Show top-3 predictions (helps you see if model is “stuck”)
                    val top3 = topK(logits, 3)

                    val displayText = top3.joinToString(" | ") { (i, s) ->
                        val name = labels.getOrElse(i) { "Unknown($i)" }
                        "$name (%.2f)".format(s)
                    }

                    val bestIdx = top3[0].first
                    val bestScore = top3[0].second

                    onPrediction(displayText, bestScore)

                } catch (e: Exception) {
                    Log.e("ORT", "Inference error", e)
                    onPrediction("Error: ${e.message}", 0f)
                } finally {
                    busy = false
                }
            }
        } finally {
            image.close() // ALWAYS close, or CameraX will stall
        }
    }

    private fun topK(arr: FloatArray, k: Int): List<Pair<Int, Float>> {
        return arr.mapIndexed { i, v -> i to v }
            .sortedByDescending { it.second }
            .take(k)
    }
}
