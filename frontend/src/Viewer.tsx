import { useCallback, useEffect, useRef, useState, type PointerEvent } from 'react'
import type { Rect } from './model'

export interface Patch {
  id: number
  rect: Rect
  status: 'pending' | 'done' | 'error'
  msiProbability?: number
  error?: string
}

interface View {
  scale: number // screen px per image px
  x: number // screen position of the image's top-left corner
  y: number
}

interface Props {
  image: ImageBitmap
  patchSize: number
  patches: Patch[]
  selectedId: number | null
  onPick: (rect: Rect) => void
}

const MAX_SCALE = 16
const DRAG_THRESHOLD = 4
const BUTTON_ZOOM = 1.5

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** Pan/zoom image viewer that lets the user click to pick a square patch. */
export default function Viewer({ image, patchSize, patches, selectedId, onPick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [view, setView] = useState<View | null>(null)
  const [hover, setHover] = useState<Rect | null>(null)
  const drag = useRef<{ startX: number; startY: number; viewX: number; viewY: number; moved: boolean } | null>(
    null,
  )
  const fittedImage = useRef<ImageBitmap | null>(null)

  const fitScale = Math.min(size.width / image.width, size.height / image.height)
  const minScale = Math.min(fitScale, 1) / 4
  const patchFits = patchSize <= image.width && patchSize <= image.height

  useEffect(() => {
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize({ width, height })
    })
    ro.observe(containerRef.current!)
    return () => ro.disconnect()
  }, [])

  const fit = useCallback(() => {
    if (!size.width || !size.height) return
    const scale = fitScale * 0.95
    setView({
      scale,
      x: (size.width - image.width * scale) / 2,
      y: (size.height - image.height * scale) / 2,
    })
  }, [image, size, fitScale])

  // Fit each newly opened image once the canvas has a size.
  useEffect(() => {
    if (fittedImage.current !== image && size.width && size.height) {
      fittedImage.current = image
      fit()
    }
  }, [image, size, fit])

  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) => {
      setView((v) => {
        if (!v) return v
        const scale = clamp(v.scale * factor, minScale, MAX_SCALE)
        const k = scale / v.scale
        return { scale, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k }
      })
    },
    [minScale],
  )

  // React's onWheel is passive, so preventDefault (to stop page scroll) needs a native listener.
  useEffect(() => {
    const canvas = canvasRef.current!
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = canvas.getBoundingClientRect()
      zoomAt(Math.exp(-e.deltaY * 0.002), e.clientX - r.left, e.clientY - r.top)
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [zoomAt])

  /** The patch centered under a screen point, clamped to stay inside the image. */
  function patchAt(sx: number, sy: number): Rect | null {
    if (!view || !patchFits) return null
    const ix = (sx - view.x) / view.scale
    const iy = (sy - view.y) / view.scale
    if (ix < 0 || iy < 0 || ix >= image.width || iy >= image.height) return null
    return {
      x: clamp(Math.round(ix - patchSize / 2), 0, image.width - patchSize),
      y: clamp(Math.round(iy - patchSize / 2), 0, image.height - patchSize),
      width: patchSize,
      height: patchSize,
    }
  }

  function localPoint(e: PointerEvent) {
    const r = canvasRef.current!.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  function onPointerDown(e: PointerEvent<HTMLCanvasElement>) {
    if (!view || e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { startX: e.clientX, startY: e.clientY, viewX: view.x, viewY: view.y, moved: false }
  }

  function onPointerMove(e: PointerEvent<HTMLCanvasElement>) {
    const d = drag.current
    if (d) {
      const dx = e.clientX - d.startX
      const dy = e.clientY - d.startY
      if (!d.moved && Math.hypot(dx, dy) > DRAG_THRESHOLD) d.moved = true
      if (d.moved) {
        setView((v) => v && { ...v, x: d.viewX + dx, y: d.viewY + dy })
        setHover(null)
        return
      }
    }
    const p = localPoint(e)
    setHover(patchAt(p.x, p.y))
  }

  function onPointerUp(e: PointerEvent<HTMLCanvasElement>) {
    const d = drag.current
    drag.current = null
    if (d && !d.moved) {
      const p = localPoint(e)
      const rect = patchAt(p.x, p.y)
      if (rect) onPick(rect)
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !view || !size.width || !size.height) return
    const dpr = window.devicePixelRatio || 1
    const w = Math.round(size.width * dpr)
    const h = Math.round(size.height * dpr)
    if (canvas.width !== w) canvas.width = w
    if (canvas.height !== h) canvas.height = h

    const ctx = canvas.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.width, size.height)
    // Show individual pixels when zoomed in past 2x.
    ctx.imageSmoothingEnabled = view.scale < 2

    // Draw only the visible part of the image.
    const sx0 = Math.max(0, -view.x / view.scale)
    const sy0 = Math.max(0, -view.y / view.scale)
    const sx1 = Math.min(image.width, (size.width - view.x) / view.scale)
    const sy1 = Math.min(image.height, (size.height - view.y) / view.scale)
    if (sx1 > sx0 && sy1 > sy0) {
      ctx.drawImage(
        image,
        sx0,
        sy0,
        sx1 - sx0,
        sy1 - sy0,
        view.x + sx0 * view.scale,
        view.y + sy0 * view.scale,
        (sx1 - sx0) * view.scale,
        (sy1 - sy0) * view.scale,
      )
    }

    const styles = getComputedStyle(canvas)
    const color = (name: string) => styles.getPropertyValue(name).trim()
    const toScreen = (r: Rect) =>
      [view.x + r.x * view.scale, view.y + r.y * view.scale, r.width * view.scale, r.height * view.scale] as const

    ctx.font = '600 12px system-ui, sans-serif'
    for (const p of patches) {
      const [x, y, w, h] = toScreen(p.rect)
      const stroke =
        p.status === 'done'
          ? color(p.msiProbability! >= 0.5 ? '--msi' : '--mss')
          : color(p.status === 'error' ? '--error' : '--muted')
      ctx.lineWidth = p.id === selectedId ? 3 : 1.5
      ctx.strokeStyle = stroke
      ctx.strokeRect(x, y, w, h)

      if (p.status === 'done' && w >= 40) {
        const text = `${(p.msiProbability! * 100).toFixed(0)}%`
        const tw = ctx.measureText(text).width
        ctx.fillStyle = stroke
        ctx.fillRect(x, y, tw + 8, 18)
        ctx.fillStyle = '#fff'
        ctx.fillText(text, x + 4, y + 13)
      }
    }

    if (hover) {
      // Dark and light dashes interleaved so the cursor shows on any tissue color.
      const [x, y, w, h] = toScreen(hover)
      ctx.lineWidth = 2
      ctx.setLineDash([6, 6])
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)'
      ctx.strokeRect(x, y, w, h)
      ctx.lineDashOffset = 6
      ctx.strokeStyle = '#fff'
      ctx.strokeRect(x, y, w, h)
      ctx.setLineDash([])
      ctx.lineDashOffset = 0
    }
  }, [image, view, size, patches, selectedId, hover])

  return (
    <div className="viewer" ref={containerRef}>
      <canvas
        ref={canvasRef}
        style={{ cursor: patchFits ? 'crosshair' : 'grab' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => (drag.current = null)}
        onPointerLeave={() => setHover(null)}
      />
      <div className="zoom-controls">
        <button
          className="icon"
          aria-label="Zoom out"
          onClick={() => zoomAt(1 / BUTTON_ZOOM, size.width / 2, size.height / 2)}
        >
          −
        </button>
        <span className="zoom-level">{view ? `${Math.round(view.scale * 100)}%` : ''}</span>
        <button
          className="icon"
          aria-label="Zoom in"
          onClick={() => zoomAt(BUTTON_ZOOM, size.width / 2, size.height / 2)}
        >
          +
        </button>
        <button className="icon wide" onClick={fit}>
          Fit
        </button>
      </div>
      {!patchFits && (
        <p className="viewer-note">Image is smaller than the patch size. Use “Analyze whole image”.</p>
      )}
    </div>
  )
}
