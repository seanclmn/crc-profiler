import { useEffect, useRef, useState } from 'react'
import { decodeImage, loadModel, predictMsi, type Rect } from './model'
import Viewer, { type Patch } from './Viewer'

// Formats the browser can decode with createImageBitmap.
const ACCEPT = 'image/png,image/jpeg,image/bmp,image/webp'

// Patch sizes offered, as multiples of the model's input size. A patch should cover the same
// physical area as a training tile, e.g. 2x for an image scanned at twice the training resolution.
const PATCH_MULTIPLES = [0.5, 1, 2, 4]

type ModelStatus =
  | { kind: 'loading' }
  | { kind: 'ready'; inputSize: number }
  | { kind: 'error'; message: string }

interface OpenImage {
  bitmap: ImageBitmap
  name: string
}

export default function App() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ kind: 'loading' })
  const [image, setImage] = useState<OpenImage | null>(null)
  const [imageError, setImageError] = useState<string | null>(null)
  const [patchSize, setPatchSize] = useState(224)
  const [patches, setPatches] = useState<Patch[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const nextId = useRef(1)

  useEffect(() => {
    loadModel().then(
      ({ meta }) => {
        setModelStatus({ kind: 'ready', inputSize: meta.imageSize })
        setPatchSize(meta.imageSize)
      },
      (err) => setModelStatus({ kind: 'error', message: errorMessage(err) }),
    )
  }, [])

  // Free the previous image's decoded pixels when another is opened.
  useEffect(() => () => image?.bitmap.close(), [image])

  async function openFile(file: File | undefined) {
    if (!file) return
    setImageError(null)
    try {
      const bitmap = await decodeImage(file)
      setImage({ bitmap, name: file.name })
      setPatches([])
      setSelectedId(null)
    } catch (err) {
      setImageError(errorMessage(err))
    }
  }

  function analyze(rect: Rect) {
    if (!image || modelStatus.kind !== 'ready') return
    const existing = patches.find(
      (p) =>
        p.rect.x === rect.x && p.rect.y === rect.y && p.rect.width === rect.width && p.rect.height === rect.height,
    )
    if (existing) {
      setSelectedId(existing.id)
      return
    }

    const id = nextId.current++
    setPatches((ps) => [...ps, { id, rect, status: 'pending' }])
    setSelectedId(id)
    const update = (fields: Partial<Patch>) =>
      setPatches((ps) => ps.map((p) => (p.id === id ? { ...p, ...fields } : p)))
    predictMsi(image.bitmap, rect).then(
      (msiProbability) => update({ status: 'done', msiProbability }),
      (err) => update({ status: 'error', error: errorMessage(err) }),
    )
  }

  const ready = modelStatus.kind === 'ready'
  const selected = patches.find((p) => p.id === selectedId)

  return (
    <main className="app">
      <header>
        <h1>CRC MSI Profiler</h1>
        <p className="subtitle">
          Open an H&amp;E image of colorectal cancer, zoom in, and click a region to estimate the
          probability of microsatellite instability (MSI). The model runs in your browser, so images
          are never uploaded.
        </p>
      </header>

      <div className="toolbar">
        <button onClick={() => inputRef.current?.click()}>Open image</button>
        <label className="field">
          Patch size
          <select value={patchSize} onChange={(e) => setPatchSize(Number(e.target.value))} disabled={!ready}>
            {ready &&
              PATCH_MULTIPLES.map((k) => (
                <option key={k} value={modelStatus.inputSize * k}>
                  {modelStatus.inputSize * k} px ({k}×)
                </option>
              ))}
          </select>
        </label>
        <button
          className="secondary"
          disabled={!image || !ready}
          onClick={() => image && analyze({ x: 0, y: 0, width: image.bitmap.width, height: image.bitmap.height })}
        >
          Analyze whole image
        </button>
        <span className={`model-status ${modelStatus.kind}`}>
          {modelStatus.kind === 'loading' && 'Loading model…'}
          {modelStatus.kind === 'ready' && 'Model ready'}
          {modelStatus.kind === 'error' && `Model failed to load: ${modelStatus.message}`}
        </span>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            openFile(e.target.files?.[0])
            e.target.value = ''
          }}
        />
      </div>

      {imageError && <p className="error">{imageError}</p>}

      <div className="workspace">
        <div
          className={`stage${dragging ? ' dragging' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            openFile(e.dataTransfer.files[0])
          }}
        >
          {image ? (
            <Viewer
              image={image.bitmap}
              patchSize={patchSize}
              patches={patches}
              selectedId={selectedId}
              onPick={analyze}
            />
          ) : (
            <button className="empty" onClick={() => inputRef.current?.click()}>
              Drop an image here, or click to choose a file
            </button>
          )}
        </div>

        <aside className="panel">
          {image && (
            <p className="image-info">
              {image.name} · {image.bitmap.width}×{image.bitmap.height} px
            </p>
          )}
          {selected && image ? (
            <PatchDetails patch={selected} image={image.bitmap} />
          ) : (
            <p className="hint">
              {image
                ? 'Scroll to zoom and drag to pan. Click to analyze the patch under the cursor.'
                : 'Open an image to get started.'}
            </p>
          )}
          {patches.length > 0 && (
            <PatchList
              patches={patches}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onClear={() => {
                setPatches([])
                setSelectedId(null)
              }}
            />
          )}
        </aside>
      </div>

      <footer>For research use only. Not a diagnostic device.</footer>
    </main>
  )
}

function PatchDetails({ patch, image }: { patch: Patch; image: ImageBitmap }) {
  const { rect } = patch
  return (
    <section className="result" aria-live="polite">
      <PatchPreview image={image} rect={rect} />
      <p className="patch-coords">
        x {rect.x}, y {rect.y} · {rect.width}×{rect.height} px
      </p>
      {patch.status === 'pending' && <p className="hint">Analyzing…</p>}
      {patch.status === 'error' && <p className="error">{patch.error}</p>}
      {patch.status === 'done' && <Probability value={patch.msiProbability!} />}
    </section>
  )
}

/** The selected patch as the model sees it. */
function PatchPreview({ image, rect }: { image: ImageBitmap; rect: Rect }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const size = 224

  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(image, rect.x, rect.y, rect.width, rect.height, 0, 0, size, size)
  }, [image, rect])

  return <canvas ref={ref} width={size} height={size} className="patch-preview" />
}

function Probability({ value }: { value: number }) {
  const pct = (value * 100).toFixed(1)
  const label = value >= 0.5 ? 'MSI' : 'MSS'
  return (
    <>
      <div className="result-header">
        <span className={`badge ${label.toLowerCase()}`}>{label}</span>
        <span className="prob">
          P(MSI) = <strong>{pct}%</strong>
        </span>
      </div>
      <div
        className="meter"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number(pct)}
        aria-label="MSI probability"
      >
        <div className="meter-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="meter-labels">
        <span>MSS</span>
        <span>MSI</span>
      </div>
    </>
  )
}

interface PatchListProps {
  patches: Patch[]
  selectedId: number | null
  onSelect: (id: number) => void
  onClear: () => void
}

function PatchList({ patches, selectedId, onSelect, onClear }: PatchListProps) {
  const done = patches.filter((p) => p.status === 'done')
  const mean = done.length ? done.reduce((sum, p) => sum + p.msiProbability!, 0) / done.length : null

  return (
    <section className="patch-list">
      <div className="patch-list-header">
        <h2>Patches ({patches.length})</h2>
        <button className="link" onClick={onClear}>
          Clear
        </button>
      </div>
      {mean !== null && done.length > 1 && (
        <p className="hint">
          Mean P(MSI) over {done.length} patches: <strong>{(mean * 100).toFixed(1)}%</strong>
        </p>
      )}
      <ol>
        {patches.map((p) => (
          <li key={p.id}>
            <button className={`patch-row${p.id === selectedId ? ' selected' : ''}`} onClick={() => onSelect(p.id)}>
              <span>#{p.id}</span>
              <span className="patch-row-coords">
                {p.rect.x}, {p.rect.y}
              </span>
              <span className={`patch-row-value ${p.status === 'done' ? (p.msiProbability! >= 0.5 ? 'msi' : 'mss') : ''}`}>
                {p.status === 'done' && `${(p.msiProbability! * 100).toFixed(1)}%`}
                {p.status === 'pending' && '…'}
                {p.status === 'error' && 'error'}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  )
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}
