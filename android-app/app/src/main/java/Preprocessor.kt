package com.example.lslclassifier

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import kotlin.math.max

object Preprocessor {

    // ✅ Matches your training (ImageNet normalization)
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    // ✅ Matches your training padding: fill=(255,255,255)
    private val padColor = Color.WHITE

    /** Pad image to square (centered) using WHITE borders (like training). */
    fun padToSquare(src: Bitmap): Bitmap {
        val size = max(src.width, src.height)
        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)

        val canvas = Canvas(out)
        canvas.drawColor(padColor)

        val left = (size - src.width) / 2f
        val top = (size - src.height) / 2f
        canvas.drawBitmap(src, left, top, null)

        return out
    }

    /** Resize to size x size (training does 256x256). */
    fun resizeSquare(src: Bitmap, size: Int): Bitmap {
        return Bitmap.createScaledBitmap(src, size, size, true)
    }

    /**
     * Convert Bitmap to FloatArray in CHW format:
     * Output length = 3*H*W, for ONNX tensor shape [1,3,H,W]
     */
    fun toCHWFloatArray(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height

        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val out = FloatArray(3 * w * h)
        var idx = 0

        for (y in 0 until h) {
            for (x in 0 until w) {
                val p = pixels[idx++]

                // Android Bitmap is ARGB; extract RGB and scale to [0,1]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f

                val i = y * w + x

                // ✅ Normalize exactly like training
                out[i] = (r - mean[0]) / std[0]             // R
                out[w * h + i] = (g - mean[1]) / std[1]     // G
                out[2 * w * h + i] = (b - mean[2]) / std[2] // B
            }
        }
        return out
    }

    /** Full pipeline: pad-to-square (white) -> resize 256 -> normalize -> CHW */
    fun preprocess(bitmap: Bitmap, size: Int = 256): FloatArray {
        val squared = padToSquare(bitmap)
        val resized = resizeSquare(squared, size)
        return toCHWFloatArray(resized)
    }
}
