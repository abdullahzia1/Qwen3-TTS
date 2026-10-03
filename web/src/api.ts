// Thin client for the voicekit v1 API. Types mirror server/app.py.

export type GenParams = {
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  subtalker_temperature?: number;
  subtalker_top_k?: number;
  subtalker_top_p?: number;
  non_streaming_mode?: boolean;
};

export type VoiceVersion = {
  version: number;
  created_at: string;
  mode: "icl" | "xvec";
  language: string;
  ref_text: string;
  ref_duration_s: number;
  chars_per_sec: number | null;
  f0_median_hz: number | null;
  params: GenParams;
  notes: string;
  warnings: string[];
};

export type Voice = { name: string; latest: number; versions: VoiceVersion[] };

export type ChunkResult = {
  idx: number;
  text: string;
  boundary: "sentence" | "paragraph" | "end";
  seed: number;
  attempts: number;
  duration_s: number;
  expected_s: number;
  status: "ok" | "suspect";
  start_s: number;
  end_s: number;
  f0_median_hz?: number | null;
  issue?: string | null;
};

export type JobStatus = "queued" | "running" | "succeeded" | "partial" | "failed";

export type TTSRequest = {
  voice: string;
  text: string;
  seed: number;
  params: GenParams;
  lexicon?: Record<string, string> | null;
  max_chars: number;
  batch_size?: number;
  target_lufs: number;
};

export type Job<Req = TTSRequest> = {
  id: string;
  kind: "tts" | "voice";
  status: JobStatus;
  request: Req;
  progress: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  result: null | {
    meta: {
      voice: string;
      duration_s: number;
      elapsed_s: number;
      rtf: number | null;
      lufs: number | null;
      params: GenParams;
      suspect_chunks: number[];
      retried_chunks: number[];
      warnings?: string[];
    };
    chunks: ChunkResult[];
    audio_version: number;
    regenerations?: { idx: number; seed: number; at: string }[];
  };
};

export type EngineState = "stopped" | "loading" | "ready" | "busy" | "stopping" | "error";

export type Engine = {
  state: EngineState;
  error: string | null;
  model: null | { name_or_path: string; device: string; dtype: string };
  idle_s: number;
  idle_timeout_s: number;
  stop_pending: boolean;
  queue_depth: number;
};

export type PreviewChunk = { text: string; boundary: string; chars: number };

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = await r.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown, headers: Record<string, string> = {}): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json", ...headers },
  body: JSON.stringify(body),
});

export const api = {
  engine: () => req<Engine>("/v1/engine"),
  startEngine: () => req<Engine>("/v1/engine/start", { method: "POST" }),
  stopEngine: () => req<Engine>("/v1/engine/stop", { method: "POST" }),

  voices: () => req<Voice[]>("/v1/voices"),
  createVoice: (form: FormData) => req<Job>("/v1/voices", { method: "POST", body: form }),
  refUrl: (name: string, version: number) => `/v1/voices/${name}/v/${version}/ref.wav`,

  previewChunks: (text: string, max_chars: number, lexicon?: Record<string, string> | null) =>
    req<{ text_normalized: string; chunks: PreviewChunk[] }>("/v1/tts/preview-chunks", json({ text, max_chars, lexicon })),
  createJob: (body: TTSRequest) => req<Job>("/v1/tts/jobs", json(body, { "Idempotency-Key": crypto.randomUUID() })),
  jobs: () => req<Job[]>("/v1/tts/jobs?limit=100"),
  job: (id: string) => req<Job>(`/v1/jobs/${id}`),
  audioUrl: (job: Job) => `/v1/tts/jobs/${job.id}/audio?v=${job.result?.audio_version ?? 0}`,
  regenerate: (id: string, idx: number) => req<Job>(`/v1/tts/jobs/${id}/chunks/${idx}/regenerate`, json({})),
};

export const isDone = (s: JobStatus) => s === "succeeded" || s === "partial" || s === "failed";
