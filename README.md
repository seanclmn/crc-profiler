# CRC MSI Profiler

Estimates the probability of microsatellite instability (MSI) from H&E histology tiles of
colorectal cancer. The model runs entirely in the browser with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), so images never leave the
user's machine and no server is needed beyond static hosting.

```
export/    One-time conversion of the PyTorch model.pt to ONNX
frontend/  Vite + React + TypeScript SPA
```

## 1. Convert the model

```sh
cd export
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python export_onnx.py ~/Downloads/model.pt
```

This writes `frontend/public/model.onnx` and `frontend/public/model.json`. The JSON file holds
the preprocessing the browser applies. The script also checks that ONNX Runtime reproduces
PyTorch's output.

`model.pt` may be TorchScript, a `state_dict`, or a fully pickled `nn.Module`. Loading a pickled
module runs arbitrary code, so the script refuses unless you pass `--trust-pickle`. Do that
only for a checkpoint you trust, ideally inside a throwaway container.

Match these options to how the model was trained:

| Option          | Default             | Meaning                                                  |
| --------------- | ------------------- | -------------------------------------------------------- |
| `--arch`        | `resnet18`          | torchvision architecture (only for `state_dict` files)   |
| `--num-classes` | `2`                 | Output classes (only for `state_dict` files)             |
| `--msi-index`   | `0`                 | Output index for MSI (ImageFolder order `MSIMUT`, `MSS`) |
| `--image-size`  | `224`               | Input resolution                                         |
| `--mean`        | `0.485,0.456,0.406` | Normalization mean (ImageNet)                            |
| `--std`         | `0.229,0.224,0.225` | Normalization std (ImageNet)                             |

A model with a single output logit is passed through a sigmoid. A multi-class model is
passed through a softmax, and the `--msi-index` output is taken as P(MSI).

## 2. Run the app

```sh
cd frontend
npm install
npm run dev   # http://localhost:5173
```

`npm run build` produces a static site in `frontend/dist/`, including the model, that can be
hosted anywhere.

Open an image, scroll to zoom, drag to pan, and click to analyze the square patch under the
cursor. Each analyzed patch stays outlined on the image with its P(MSI), and the side panel
shows the exact crop the model saw plus the mean over all patches.

Set the patch size so a patch covers the same tissue area as a training tile. At 1× (e.g.
224 px for a 224 px model), the crop is fed to the model unchanged and matches PyTorch exactly.
Other sizes are resized with the browser canvas, whose resampling differs slightly from
torchvision's, so their probabilities can shift by a point or two.

For research use only. Not a diagnostic device.
