import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, isDone, type GenParams, type Job, type Voice } from "../api";
import { Waveform, fmt, type WaveformHandle } from "../components/Waveform";

type Settings = {
  seed: number;
  max_chars: number;
  target_lufs: number;
  params: GenParams;
};

const DEFAULTS: Settings = { seed: 0, max_chars: 400, target_lufs: -16, params: {} };

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export function Studio({ jobId, onJob }: { jobId: string | null; onJob: (id: string) => void }) {
  const qc = useQueryClient();
  const voices = useQuery({ queryKey: ["voices"], queryFn: api.voices });
  const [voice, setVoice] = useState("");
  const [text, setText] = useState("");
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [showSettings, setShowSettings] = useState(false);

  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data && isDone(q.state.data.status) ? false : 1000),
  });

  // Opening a job from History loads its script and settings back into the editor.
  const loadedFor = useRef<string | null>(null);
  useEffect(() => {
    const j = job.data;
    if (!j || loadedFor.current === j.id) return;
    loadedFor.current = j.id;
    setVoice(j.request.voice);
    setText(j.request.text);
    setSettings({ seed: j.request.seed, max_chars: j.request.max_chars, target_lufs: j.request.target_lufs, params: j.request.params });
  }, [job.data]);

  const voiceList = voices.data ?? [];
  useEffect(() => {
    if (!voice && voiceList.length) setVoice(`${voiceList[0].name}/v${voiceList[0].latest}`);
  }, [voice, voiceList]);
  const selected = useMemo(() => findVersion(voiceList, voice), [voiceList, voice]);
  const effective: GenParams = { ...(selected?.params ?? {}), ...settings.params };

  const debouncedText = useDebounced(text, 400);
  const preview = useQuery({
    queryKey: ["preview", debouncedText, settings.max_chars],
    queryFn: () => api.previewChunks(debouncedText, settings.max_chars),
    enabled: debouncedText.trim().length > 0,
  });

  const generate = useMutation({
    mutationFn: () => api.createJob({ voice, text, ...settings }),
    onSuccess: (j) => {
      loadedFor.current = j.id;
      qc.setQueryData(["job", j.id], j);
      onJob(j.id);
    },
  });

  const running = !!job.data && !isDone(job.data.status);
  const chars = text.length;
  const estSeconds = selected?.chars_per_sec ? chars / selected.chars_per_sec : null;

  return (
    <div className="studio">
      <section className="panel editor">
        <div className="row">
          <label className="grow">
            <span className="label">Voice</span>
            <select value={voice} onChange={(e) => setVoice(e.target.value)}>
              {voiceList.length === 0 && <option value="">No voices yet — create one in Voices</option>}
              {voiceList.flatMap((v) =>
                [...v.versions].reverse().map((ver) => (
                  <option key={`${v.name}/v${ver.version}`} value={`${v.name}/v${ver.version}`}>
                    {v.name} · v{ver.version}
                    {ver.version === v.latest ? " (latest)" : ""}
                  </option>
                )),
              )}
            </select>
          </label>
          <button className="ghost" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
            Settings {showSettings ? "▴" : "▾"}
          </button>
        </div>

        {showSettings && (
          <SettingsPanel settings={settings} effective={effective} onChange={setSettings} voiceDefaults={selected?.params} />
        )}

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your script. Blank lines mark paragraphs (longer pause)."
          rows={12}
        />
        <div className="row meta">
          <span className="muted small">
            {chars.toLocaleString()} chars
            {preview.data && ` · ${preview.data.chunks.length} chunks`}
            {estSeconds ? ` · ~${fmt(estSeconds)} est.` : null}
          </span>
          <button
            className="primary"
            disabled={!voice || !text.trim() || generate.isPending || running}
            onClick={() => generate.mutate()}
          >
            {running ? "Generating…" : "Generate"}
          </button>
        </div>
        {generate.error && <p className="error">{String(generate.error.message)}</p>}

        {preview.data && preview.data.chunks.length > 1 && (
          <details className="chunk-preview">
            <summary className="small muted">Where the script will be split</summary>
            <ol>
              {preview.data.chunks.map((c, i) => (
                <li key={i}>
                  <span>{c.text}</span>
                  <span className="muted small mono"> {c.chars}c · {c.boundary}</span>
                </li>
              ))}
            </ol>
          </details>
        )}
      </section>

      <section className="panel output">{job.data ? <Result job={job.data} /> : <Empty />}</section>
    </div>
  );
}

function findVersion(voices: Voice[], id: string) {
  const [name, ver] = id.split("/v");
  const v = voices.find((x) => x.name === name);
  return v?.versions.find((x) => String(x.version) === ver);
}

function SettingsPanel({
  settings,
  effective,
  voiceDefaults,
  onChange,
}: {
  settings: Settings;
  effective: GenParams;
  voiceDefaults?: GenParams;
  onChange: (s: Settings) => void;
}) {
  const setParam = (k: keyof GenParams, v: number | boolean) => onChange({ ...settings, params: { ...settings.params, [k]: v } });
  const num = (k: keyof GenParams, label: string, min: number, max: number, step: number, hint: string) => (
    <label title={hint}>
      <span className="label">
        {label} <span className="mono">{String(effective[k] ?? "—")}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={Number(effective[k] ?? min)}
        onChange={(e) => setParam(k, Number(e.target.value))} />
    </label>
  );
  return (
    <div className="settings">
      {num("temperature", "Temperature", 0.1, 1.2, 0.05, "Prosody randomness. Lower = steadier, risk of flatness.")}
      {num("subtalker_temperature", "Timbre temperature", 0.1, 1.2, 0.05, "Randomness of fine acoustic detail.")}
      {num("top_k", "Top-k", 1, 100, 1, "Sampling pool size.")}
      {num("repetition_penalty", "Repetition penalty", 1.0, 1.3, 0.01, "Discourages loops.")}
      <label>
        <span className="label">Seed</span>
        <input type="number" value={settings.seed} onChange={(e) => onChange({ ...settings, seed: Number(e.target.value) })} />
      </label>
      <label>
        <span className="label">Max chunk chars</span>
        <input type="number" min={80} max={1200} step={20} value={settings.max_chars}
          onChange={(e) => onChange({ ...settings, max_chars: Number(e.target.value) })} />
      </label>
      <label>
        <span className="label">Loudness (LUFS)</span>
        <input type="number" min={-30} max={-8} step={0.5} value={settings.target_lufs}
          onChange={(e) => onChange({ ...settings, target_lufs: Number(e.target.value) })} />
      </label>
      <label className="check">
        <input type="checkbox" checked={!!effective.non_streaming_mode}
          onChange={(e) => setParam("non_streaming_mode", e.target.checked)} />
        <span>Non-streaming text layout</span>
      </label>
      <button className="ghost small" disabled={Object.keys(settings.params).length === 0}
        onClick={() => onChange({ ...settings, params: {} })}
        title={voiceDefaults ? JSON.stringify(voiceDefaults) : ""}>
        Reset to voice defaults
      </button>
    </div>
  );
}

function Result({ job }: { job: Job }) {
  const qc = useQueryClient();
  const wave = useRef<WaveformHandle>(null);
  const [now, setNow] = useState(0);
  const regen = useMutation({
    mutationFn: (idx: number) => api.regenerate(job.id, idx),
    onSuccess: (j) => qc.setQueryData(["job", j.id], j),
  });

  if (job.status === "failed") return <p className="error">Failed: {job.error}</p>;
  if (!job.result) {
    return (
      <div className="progress-box">
        <p>{job.status === "queued" ? "Queued — waiting for the model…" : "Generating…"}</p>
        <div className="progress"><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
      </div>
    );
  }

  const { meta, chunks } = job.result;
  const busy = !isDone(job.status);
  return (
    <>
      <div className="row">
        <h2 className="grow">{meta.voice}</h2>
        <span className={`badge ${job.status}`}>{busy ? "regenerating…" : job.status}</span>
        <a className="button ghost" href={api.audioUrl(job)} download={`${job.id}.wav`}>Download WAV</a>
      </div>
      <Waveform ref={wave} url={api.audioUrl(job)} onTime={setNow} />
      <div className="stats small muted mono">
        {fmt(meta.duration_s)} · {meta.lufs} LUFS · RTF {meta.rtf ?? "—"} · {chunks.length} chunks
        {meta.retried_chunks.length > 0 && ` · retried ${meta.retried_chunks.join(", ")}`}
      </div>
      <ol className="chunks">
        {chunks.map((c) => {
          const active = now >= c.start_s && now < c.end_s;
          const ratio = c.duration_s / c.expected_s;
          return (
            <li key={c.idx} className={`${active ? "active" : ""} ${c.status}`}>
              <button className="ghost icon" onClick={() => wave.current?.playRange(c.start_s, c.end_s)} aria-label={`Play chunk ${c.idx + 1}`}>▶</button>
              <div className="grow">
                <p>{c.text}</p>
                <span className="small muted mono">
                  #{c.idx + 1} · {c.duration_s.toFixed(1)}s ({Math.round(ratio * 100)}% of expected) · seed {c.seed}
                  {c.attempts > 1 && ` · ${c.attempts} attempts`}
                </span>
              </div>
              {c.status === "suspect" && <span className="badge partial" title="Length outside the expected range">check</span>}
              <button className="ghost small" disabled={busy || regen.isPending} onClick={() => regen.mutate(c.idx)}>
                Regenerate
              </button>
            </li>
          );
        })}
      </ol>
      {regen.error && <p className="error">{String(regen.error.message)}</p>}
    </>
  );
}

function Empty() {
  return (
    <div className="empty muted">
      <p>Generated audio appears here, split into chunks you can play and regenerate one at a time.</p>
    </div>
  );
}
