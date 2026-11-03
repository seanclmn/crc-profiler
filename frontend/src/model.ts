import * as ort from 'onnxruntime-web'

// Written by export/export_onnx.py alongside model.onnx.
export interface ModelMeta {
  inputName: string
  outputName: string
  imageSize: number
  mean: [number, number, number]
  std: [number, number, number]
  activation: 'softmax' | 'sigmoid'
  msiIndex: number
}

interface LoadedModel {
  session: ort.InferenceSession
  meta: ModelMeta
}

/** A region of the source image, in source pixels. */
export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

const MODEL_URL = `${import.meta.env.BASE_URL}model.onnx`
const META_URL = `${import.meta.env.BASE_URL}model.json`

let loading: Promise<LoadedModel> | null = null

export function loadModel(): Promise<LoadedModel> {
  loading ??= (async () => {
    const metaRes = await fetch(META_URL)
    if (!metaRes.ok) throw new Error(`Could not load ${META_URL} (${metaRes.status})`)
    const meta: ModelMeta = await metaRes.json()
    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
    })
    return { session, meta }
  })().catch((err) => {
    loading = null // allow a retry
    throw err
  })
  return loading
}

export async function decodeImage(file: File): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(file)
  } catch {
    throw new Error('Could not decode image')
  }
}

/** Crop rect from source, resize to the model input size, and convert to normalized NCHW. */
function preprocess(source: CanvasImageSource, rect: Rect, meta: ModelMeta): ort.Tensor {
  const size = meta.imageSize
  const canvas = new OffscreenCanvas(size, size)
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(source, rect.x, rect.y, rect.width, rect.height, 0, 0, size, size)

  const { data } = ctx.getImageData(0, 0, size, size)
  const plane = size * size
  const chw = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      chw[c * plane + i] = (data[i * 4 + c] / 255 - meta.mean[c]) / meta.std[c]
    }
  }
  return new ort.Tensor('float32', chw, [1, 3, size, size])
}

async function infer(source: CanvasImageSource, rect: Rect): Promise<number> {
  const { session, meta } = await loadModel()
  const input = preprocess(source, rect, meta)
  const output = await session.run({ [meta.inputName]: input })
  const logits = Array.from(output[meta.outputName].data as Float32Array)

  if (meta.activation === 'sigmoid') return 1 / (1 + Math.exp(-logits[0]))
  const max = Math.max(...logits)
  const exps = logits.map((x) => Math.exp(x - max))
  return exps[meta.msiIndex] / exps.reduce((a, b) => a + b, 0)
}

// An ONNX Runtime Web session can only run one inference at a time, so queue them.
let queue: Promise<unknown> = Promise.resolve()

/** Return P(MSI) for the given region of a histology image. */
export function predictMsi(source: CanvasImageSource, rect: Rect): Promise<number> {
  const run = queue.then(() => infer(source, rect))
  queue = run.catch(() => {})
  return run
}
