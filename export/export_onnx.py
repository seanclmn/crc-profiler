"""Convert a PyTorch MSI classifier (model.pt) to ONNX for in-browser inference.

Writes <output>.onnx plus a sidecar <output>.json describing the preprocessing the
browser must apply, then checks that ONNX Runtime reproduces PyTorch's output.

The defaults assume a typical Kaggle CRC MSI-vs-MSS tile classifier (e.g. trained on
the Kather TCGA tiles): 224x224 RGB input, ImageNet normalization, and two classes in
ImageFolder's alphabetical order ["MSIMUT", "MSS"], so MSI is index 0.

    python export_onnx.py ~/Downloads/model.pt
"""

import argparse
import io
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "frontend" / "public" / "model.onnx"


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, help="Path to model.pt")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--arch", default="resnet18", help="torchvision architecture, for state_dict checkpoints")
    p.add_argument("--num-classes", type=int, default=2, help="Output classes, for state_dict checkpoints")
    p.add_argument("--msi-index", type=int, default=0, help="Output index meaning MSI (multi-class models)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--mean", type=floats, default=[0.485, 0.456, 0.406])
    p.add_argument("--std", type=floats, default=[0.229, 0.224, 0.225])
    p.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Allow loading a fully pickled nn.Module. Unpickling runs arbitrary code, so only "
        "use this on checkpoints you trust, ideally inside a throwaway container.",
    )
    return p.parse_args()


def strip_prefix(state_dict: dict) -> dict:
    # Checkpoints saved from nn.DataParallel prefix every key with "module.".
    if all(k.startswith("module.") for k in state_dict):
        return OrderedDict((k[len("module."):], v) for k, v in state_dict.items())
    return state_dict


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    path = str(args.checkpoint)
    try:
        return torch.jit.load(path, map_location="cpu")
    except RuntimeError:
        pass  # Not TorchScript.

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not args.trust_pickle:
            raise SystemExit(
                f"{path} is not a plain state_dict, it is likely a pickled nn.Module.\n"
                "Re-run with --trust-pickle if you trust its source.\n"
                f"({type(e).__name__}: {str(e).splitlines()[0]})"
            )
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        state_dict = obj.get("state_dict") or obj.get("model_state_dict") or obj
        model = torchvision.models.get_model(args.arch, weights=None, num_classes=args.num_classes)
        model.load_state_dict(strip_prefix(state_dict))
        return model
    raise SystemExit(f"Unsupported checkpoint type: {type(obj).__name__}")


def main() -> None:
    args = parse_args()
    model = load_model(args).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    with torch.inference_mode():
        expected = model(dummy)
    if isinstance(expected, (tuple, list)):
        expected = expected[0]
    num_outputs = expected.flatten().numel()
    activation = "sigmoid" if num_outputs == 1 else "softmax"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = dict(input_names=["input"], output_names=["logits"])
    if isinstance(model, torch.jit.ScriptModule):
        # The dynamo exporter can't trace TorchScript, so use the legacy exporter.
        buf = io.BytesIO()
        torch.onnx.export(model, (dummy,), buf, dynamo=False, **names)
        proto = onnx.load_from_string(buf.getvalue())
    else:
        proto = torch.onnx.export(model, (dummy,), dynamo=True, **names).model_proto
    # Keep weights inside the .onnx file so the browser fetches a single asset.
    onnx.save(proto, str(args.output), save_as_external_data=False)

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(actual - expected.numpy()).max())
    if max_diff > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch by {max_diff:.2e}")

    meta = {
        "inputName": "input",
        "outputName": "logits",
        "imageSize": args.image_size,
        "mean": args.mean,
        "std": args.std,
        "activation": activation,
        "msiIndex": 0 if activation == "sigmoid" else args.msi_index,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.1f} MB), {num_outputs} output(s), {activation}")
    print(f"Wrote {args.output.with_suffix('.json')}")
    print(f"ONNX matches PyTorch (max abs diff {max_diff:.1e})")


if __name__ == "__main__":
    main()
