# INT8 quantization: the theory

This page explains what happens when a network runs in INT8 on TensorRT, and why it has to be *calibrated*. The fieldbench implementation is in [int8-calibration-in-fieldbench.md](int8-calibration-in-fieldbench.md).

## Why INT8

A network trained in FP32 can usually run with 8-bit integers for weights and activations at almost the same accuracy. The payoff:

- **Speed.** INT8 tensor-core math has about twice the throughput of FP16 on Ampere, the Orin Nano's GPU.
- **Memory traffic.** Each value is a quarter the size of FP32 and half the size of FP16. On a shared-memory SoC like the Orin, bandwidth is often what limits speed.
- **Energy.** Less data moved and simpler arithmetic mean fewer joules per frame. In fieldbench's Phase 1 sweep, FP32 → INT8 cut YOLO11n's energy per frame by 2.6×.

Handheld NPUs such as the QCS6490's Hexagon are built around 8-bit math, which is why Qualcomm's published numbers are `w8a8` (8-bit weights and activations). So INT8 is also the like-for-like comparison.

## Mapping real numbers to 8 bits

TensorRT uses **symmetric, uniform** quantization. Each tensor (or each weight channel) gets one scale `s`, and the zero point is fixed at 0:

```
s     = amax / 127                       amax: the largest magnitude we decide to represent
q     = clamp(round(x / s), -127, 127)   quantize  (stored as int8)
x_hat = q * s                            dequantize (what the next layer effectively sees)
```

Everything hinges on **amax**, the clipping threshold:

- Values inside `[-amax, amax]` get a **rounding error** of at most `s / 2`.
- Values beyond it are **clipped** to ±amax, which can be a much larger error.

A large amax avoids clipping but makes every step coarser. A small amax gives fine steps but clips the tails. Calibration is the job of picking amax for every activation tensor.

There are only 255 levels. If one tensor holds values from 0 to 700 (pixel coordinates), each step is about 5.5 units, and anything that lives between 0 and 1 in that tensor collapses to a few levels.

## Weights vs activations

**Weights** are known at build time, so their scales need no data. TensorRT quantizes convolution and fully-connected weights **per output channel**: each filter gets its own scale from its own largest magnitude. That keeps one large filter from coarsening all the others.

**Activations** depend on the input image, so their range can only be measured by running data through the network. TensorRT uses **one scale per activation tensor** (per tensor, not per channel). Measuring those ranges on representative inputs is what **calibration** means.

Without calibration, TensorRT has no idea what range each activation covers. `trtexec --int8` with no calibration data falls back to placeholder ranges. The engine builds and runs at real INT8 speed, so it is fine for **timing**, but its outputs are meaningless.

## Calibration algorithms

During calibration TensorRT runs the FP32 network on a calibration set, records the activation statistics of every tensor, and turns them into an amax per tensor. The calibrators differ in how they turn statistics into a threshold:

| Calibrator (TensorRT class) | How amax is chosen | Good at | Risk |
|---|---|---|---|
| **MinMax** (`IInt8MinMaxCalibrator`) | The largest absolute value seen anywhere in the calibration set | Tensors whose large values matter and must not be clipped | One outlier inflates the step size for the whole tensor |
| **Entropy** (`IInt8EntropyCalibrator2`, TensorRT's default recommendation for CNNs) | Builds a histogram of \|x\|, then picks the threshold whose quantized distribution is closest to the original, measured by KL divergence | Bell-shaped activations with long, rare tails (typical CNN feature maps): clips the tails for finer steps in the bulk | Clips values that are rare but important |
| **Percentile** (`IInt8LegacyCalibrator`) | A high percentile of \|x\| (for example 99.99%) | A simple, tunable compromise | The percentile is a guess per model |

MinMax never clips but may waste resolution. Entropy and percentile trade a little clipping for resolution. Which one wins depends on the network, and it should be **measured on accuracy**, not assumed.

### What makes a good calibration set

- **Same preprocessing as inference.** The calibration batches must be produced by exactly the resize, padding, channel order and normalization the deployed pipeline uses. Otherwise the measured ranges belong to a different input distribution.
- **Same content as deployment.** Images should look like what the device will see: lighting, blur, object sizes. Never use the test set, or the accuracy you report afterwards is optimistic.
- **Enough images to cover the range.** A few hundred is typical (NVIDIA's guidance for ImageNet classifiers is about 500). MinMax is more sensitive to *which* images are included than to how many, since one extreme frame sets the scale.

## Implicit vs explicit quantization

TensorRT has two ways to run INT8:

- **Implicit quantization (calibration):** the method above. You enable INT8 (and usually FP16) in the builder config and attach a calibrator. TensorRT then picks each layer's precision **by speed**: a layer runs in INT8 only if an INT8 kernel exists and is faster, otherwise in FP16 or FP32. You don't directly control which layers end up INT8. Since TensorRT 10.1 the calibrator API is deprecated in favor of explicit quantization, but it still works in 10.3, the version on this Jetson.
- **Explicit quantization (Q/DQ):** the ONNX model itself carries QuantizeLinear/DequantizeLinear nodes with fixed scales. These come from post-training quantization tooling (for example NVIDIA TensorRT Model Optimizer) or from quantization-aware training (QAT), where the network is fine-tuned with simulated quantization so it learns to tolerate it. TensorRT then follows the Q/DQ placement **exactly**. This allows per-channel activation tricks, keeping sensitive layers in higher precision on purpose, and accuracy recovery through QAT. It's the route to take when calibration alone loses too much accuracy.

## Mixed precision is normal

An INT8 engine is never all INT8. Layers without INT8 kernels, layers where INT8 isn't faster, and reformatting at tensor boundaries run in FP16 (if enabled) or FP32. Enabling FP16 alongside INT8 matters: without it, every fallback layer would run in FP32.

Detection heads are the usual sensitive spot. Their final outputs mix quantities with very different ranges, such as box coordinates in pixels and scores between 0 and 1. If such a tensor were quantized with a single scale, either the coordinates clip or the scores lose nearly all resolution. In practice TensorRT often keeps these tail layers in FP16. Whether it did is visible in the engine's per-layer precision, and the accuracy check is the final arbiter.

## Further reading

- NVIDIA TensorRT Developer Guide: "Working with Quantized Types" (implicit vs explicit quantization, calibrators, the calibration cache)
- H. Wu et al., *Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation*, NVIDIA, 2020 (arXiv:2004.09602): a systematic comparison of max, entropy and percentile calibration across CNNs, transformers and detectors
- S. Migacz, *8-bit Inference with TensorRT*, GTC 2017: where the entropy (KL divergence) calibrator comes from
