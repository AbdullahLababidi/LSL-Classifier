package com.example.lslclassifier

import android.content.Context
import ai.onnxruntime.*
import java.nio.FloatBuffer

class OrtClassifier(
    private val context: Context,
    private val modelAssetName: String,
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open(modelAssetName).readBytes()
        val opts = OrtSession.SessionOptions().apply {
            // Optional performance tweaks:
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
            // If you later want NNAPI (device-dependent):
            // addNnapi()
        }
        session = env.createSession(modelBytes, opts)
    }

    fun close() {
        session.close()
        env.close()
    }

    /**
     * For dual-input model:
     * inputA and inputB are CHW float arrays for [1,3,256,256]
     *
     * Returns logits/probabilities as FloatArray length = numClasses
     */
    fun runDual(inputA: FloatArray, inputB: FloatArray): FloatArray {
        val inputNames = session.inputNames.toList()

        // Many dual-input ONNX models have 2 inputs. We’ll map them by order.
        if (inputNames.size < 2) {
            throw IllegalStateException("Model has <2 inputs. Found: $inputNames")
        }

        val shape = longArrayOf(1, 3, 256, 256)

        val tensorA = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputA), shape)
        val tensorB = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputB), shape)

        tensorA.use { a ->
            tensorB.use { b ->
                val inputs = mapOf(
                    inputNames[0] to a,
                    inputNames[1] to b
                )
                session.run(inputs).use { results ->
                    // Assume first output is [1,10] (or [10])
                    val out = results[0].value
                    return when (out) {
                        is Array<*> -> {
                            // e.g., Array<FloatArray> with shape [1][10]
                            @Suppress("UNCHECKED_CAST")
                            (out as Array<FloatArray>)[0]
                        }
                        is FloatArray -> out
                        else -> throw IllegalStateException("Unexpected output type: ${out?.javaClass}")
                    }
                }
            }
        }
    }
}
