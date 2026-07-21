import { useEffect, useRef, useState } from "react"

const CURSOR_ICON = (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18">
    <path d="M3 2.5L14.5 9L3 15.5Z" fill="#F38B3D" />
    <path d="M3 2.5L14.5 9L3 15.5Z" stroke="#FFB067" strokeWidth=".6" />
  </svg>
)

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

export function CursorOverlay() {
  const ref = useRef<HTMLDivElement>(null)
  const mouse = useRef({ x: 0, y: 0 })
  const pos = useRef({ x: 0, y: 0 })
  const addr = useRef(0x00401020)
  const prevLabel = useRef("")

  const [visible, setVisible] = useState(false)
  const [label, setLabel] = useState("")
  const [clickFlash, setClickFlash] = useState(false)
  const [hovering, setHovering] = useState(false)

  useEffect(() => {
    let moved = false

    const onMove = (e: MouseEvent) => {
      mouse.current = { x: e.clientX, y: e.clientY }
      if (!moved) {
        moved = true
        setVisible(true)
      }
    }

    const onLeave = () => setVisible(false)
    const onEnter = () => setVisible(true)

    document.addEventListener("mousemove", onMove)
    document.addEventListener("mouseleave", onLeave)
    document.addEventListener("mouseenter", onEnter)

    return () => {
      document.removeEventListener("mousemove", onMove)
      document.removeEventListener("mouseleave", onLeave)
      document.removeEventListener("mouseenter", onEnter)
    }
  }, [])

  useEffect(() => {
    const el = ref.current
    if (!el) return
    let id: number
    const tick = () => {
      pos.current.x = lerp(pos.current.x, mouse.current.x, 0.15)
      pos.current.y = lerp(pos.current.y, mouse.current.y, 0.15)
      el.style.left = `${pos.current.x + 12}px`
      el.style.top = `${pos.current.y + 10}px`
      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      addr.current += 0x4
      if (addr.current > 0x00401040) addr.current = 0x00401020
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const onOver = (e: MouseEvent) => {
      const t = e.target as HTMLElement
      const el = t.closest("[data-cursor]") as HTMLElement | null
      if (el) {
        setLabel(el.dataset.cursor || "")
        setHovering(true)
      } else {
        setLabel("")
        setHovering(false)
      }
    }
    document.addEventListener("mouseover", onOver)
    return () => document.removeEventListener("mouseover", onOver)
  }, [])

  useEffect(() => {
    const onDown = () => {
      prevLabel.current = label
      setClickFlash(true)
      setTimeout(() => setClickFlash(false), 120)
    }
    document.addEventListener("mousedown", onDown)
    return () => document.removeEventListener("mousedown", onDown)
  }, [label])

  const hex = `0x${addr.current.toString(16).toUpperCase().padStart(8, "0")}`
  const display = clickFlash ? "step" : (label || hex)

  return (
    <div
      ref={ref}
      style={{
        position: "fixed",
        pointerEvents: "none",
        zIndex: 99999,
        fontFamily: "'JetBrains Mono', 'IBM Plex Mono', monospace",
        fontSize: "13px",
        color: "#F7E8DB",
        background: "#18110D",
        border: "1px solid rgba(255,145,70,.18)",
        borderRadius: "4px",
        padding: "4px 8px",
        display: "flex",
        alignItems: "center",
        gap: "6px",
        whiteSpace: "nowrap",
        opacity: visible ? 1 : 0,
        transform: visible
          ? `translateY(0px) scale(${hovering ? 1.04 : 1})`
          : "translateY(4px) scale(1)",
        transition: "opacity 150ms ease, transform 150ms ease",
        boxShadow: hovering ? "0 0 12px rgba(243,139,61,.3)" : "none",
        left: 0,
        top: 0,
      }}
    >
      {CURSOR_ICON}
      <span>{display}</span>
    </div>
  )
}
