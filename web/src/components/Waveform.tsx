import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

export type WaveformHandle = { playRange: (start: number, end: number) => void };

type Props = { url: string; height?: number; onTime?: (t: number) => void };

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const Waveform = forwardRef<WaveformHandle, Props>(function Waveform({ url, height = 96, onTime }, ref) {
  const box = useRef<HTMLDivElement>(null);
  const ws = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;

  useEffect(() => {
    if (!box.current) return;
    const w = WaveSurfer.create({
      container: box.current,
      url,
      height,
      waveColor: css("--wave"),
      progressColor: css("--accent"),
      cursorColor: css("--text"),
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
    });
    w.on("ready", (d) => setDuration(d));
    w.on("play", () => setPlaying(true));
    w.on("pause", () => setPlaying(false));
    w.on("finish", () => setPlaying(false));
    w.on("timeupdate", (t) => {
      setTime(t);
      onTimeRef.current?.(t);
    });
    ws.current = w;
    return () => {
      w.destroy();
      ws.current = null;
    };
  }, [url, height]);

  useImperativeHandle(ref, () => ({
    playRange: (start, end) => void ws.current?.play(start, end),
  }));

  return (
    <div className="waveform">
      <button className="play" onClick={() => ws.current?.playPause()} aria-label={playing ? "Pause" : "Play"}>
        {playing ? "❚❚" : "▶"}
      </button>
      <div className="wave" ref={box} />
      <span className="time mono">
        {fmt(time)} / {fmt(duration)}
      </span>
    </div>
  );
});

export const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
