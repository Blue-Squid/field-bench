"""Benchmark models and published handheld-device reference latencies.

Most references are batch-1 latencies from Qualcomm AI Hub model cards, measured on the
QCS6490 (the SoC in Zebra TC53/TC58-class rugged handhelds). They are pure model
inference on the Hexagon NPU, comparable to this tool's gpu_ms, not end-to-end.
"""

QCS6490 = "Qualcomm QCS6490 (Zebra TC53/TC58 class)"
AIHUB = "https://huggingface.co/qualcomm/{}"


def _qcs(runtime, precision, ms, card):
    return {"device": QCS6490, "runtime": runtime, "precision": precision,
            "unit": "NPU", "latency_ms": ms, "source": AIHUB.format(card)}


MODELS = {
    "yolo11n": {
        "onnx": "models/yolo11n.onnx",
        "task": "detection",
        "workload": "General object detection (80 COCO classes)",
        "input": "1x3x640x640",
        "params_m": 2.64,
        "license": "AGPL-3.0 (Ultralytics)",
        "references": [
            _qcs("TFLite", "w8a8", 4.098, "YOLOv11-Detection"),
            _qcs("ONNX", "w8a16", 20.585, "YOLOv11-Detection"),
        ],
    },
    "mobilenetv3l": {
        "onnx": "models/mobilenetv3l.onnx",
        "task": "classification",
        "workload": "Image classification / embedding backbone",
        "input": "1x3x224x224",
        "params_m": 5.47,
        "license": "BSD-3-Clause (torchvision)",
        "references": [
            _qcs("TFLite", "w8a8", 1.165, "MobileNet-v3-Large"),
            _qcs("ONNX", "w8a8", 1.326, "MobileNet-v3-Large"),
            _qcs("QNN_DLC", "w8a8", 1.653, "MobileNet-v3-Large"),
            _qcs("ONNX", "w8a16", 2.67, "MobileNet-v3-Large"),
            _qcs("QNN_DLC", "w8a16", 3.051, "MobileNet-v3-Large"),
        ],
    },
}
