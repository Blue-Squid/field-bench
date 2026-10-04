"""Benchmark models and published handheld-device reference latencies.

Most references are batch-1 latencies from Qualcomm AI Hub model cards, measured on the
QCS6490 (the SoC in Zebra TC53/TC58-class rugged handhelds). They are pure model
inference on the Hexagon NPU, comparable to this tool's gpu_ms, not end-to-end.

Zebra references are the AI Data Capture SDK's own barcode model on a TC53 (QCS6490 DSP).
Zebra's "detection time" probably includes pre/postprocessing, so it sits between our
gpu_ms and the pipeline's preprocess+infer+postprocess; "detection + decode" is end to end.
"""

QCS6490 = "Qualcomm QCS6490 (Zebra TC53/TC58 class)"
AIHUB = "https://huggingface.co/qualcomm/{}"


ZEBRA_BARCODE = "https://techdocs.zebra.com/ai-datacapture/latest/model/barcode-localizer/"


def _zebra(ms, what, source=ZEBRA_BARCODE):
    return {"device": "Zebra TC53 (QCS6490)", "runtime": "Zebra AI Data Capture SDK", "precision": "?",
            "unit": "DSP", "latency_ms": ms, "covers": what, "source": source}


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

# Barcode detector: YOLO11n fine-tuned on host/make_barcodes.py scenes, at Zebra's three input sizes.
# Zebra's model is its own (unpublished architecture); only the input size and job match.
for _size, _det, _dd in [(640, 22, 57), (1280, 59, 94), (1600, 89, 124)]:
    MODELS[f"barcode_yolo11n_{_size}"] = {
        "onnx": f"models/barcode_yolo11n_{_size}.onnx",
        "task": "detection",
        "workload": "Barcode localization (1D + 2D), synthetic-trained",
        "input": f"1x3x{_size}x{_size}",
        "params_m": 2.58,
        "license": "AGPL-3.0 (Ultralytics)",
        "references": [_zebra(_det, "detection")],
        "pipeline_references": [_zebra(_dd, "detection + decode")],
    }


# OCR: PP-OCRv5 mobile (PaddleOCR, Apache-2.0, ONNX via RapidOCR). Zebra's TextOCR model times
# are whole-pipeline (detect + recognize) on a TC53 at each input size.
ZEBRA_OCR = "https://techdocs.zebra.com/ai-datacapture/latest/model/textocr/"
for _size, _ms in [(640, 110), (1280, 180), (1600, 270), (2560, 480)]:
    MODELS[f"ppocr5_det_{_size}"] = {
        "onnx": f"models/ppocr5_det_{_size}.onnx",
        "task": "text detection",
        "workload": "Text detection (DB, PP-OCRv5 mobile)",
        "input": f"1x3x{_size}x{_size}",
        "params_m": 4.6,
        "license": "Apache-2.0 (PaddleOCR)",
        "references": [],
        "pipeline_references": [_zebra(_ms, "text detection + recognition", ZEBRA_OCR)],
    }
MODELS["ppocr5_rec_en"] = {
    "onnx": "models/ppocr5_rec_en.onnx",
    "task": "text recognition",
    "workload": "Text-line recognition (SVTR-LCNet, PP-OCRv5 mobile, English)",
    "input": "8x3x48x640",
    "params_m": 7.5,
    "license": "Apache-2.0 (PaddleOCR)",
    "references": [],
}
# Context only (different model): EasyOCR on the QCS6490, from Qualcomm AI Hub, input 608x800.
EASYOCR_QCS6490 = [_qcs("TFLite", "w8a8", 52.253, "EasyOCR") | {"covers": "detector (CRAFT), NPU"},
                   _qcs("TFLite", "w8a8", 181.161, "EasyOCR") | {"covers": "recognizer, runs on CPU", "unit": "CPU"}]