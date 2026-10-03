import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, isDone } from "../api";

const LANGUAGES = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"];

export function Voices() {
  const voices = useQuery({ queryKey: ["voices"], queryFn: api.voices });
  return (
    <div className="voices">
      <section className="panel">
        <h2>Voices</h2>
        {voices.data?.length === 0 && <p className="muted">No voices yet. Create one from a reference clip.</p>}
        <ul className="voice-list">
          {voices.data?.map((v) => (
            <li key={v.name}>
              <h3>{v.name}</h3>
              {[...v.versions].reverse().map((ver) => (
                <div key={ver.version} className="version">
                  <div className="row">
                    <strong className="mono">v{ver.version}</strong>
                    <span className="small muted">
                      {new Date(ver.created_at).toLocaleString()} · {ver.language} · {ver.mode} · {ver.ref_duration_s}s ref
                      {ver.chars_per_sec && ` · ${ver.chars_per_sec} chars/s`}
                    </span>
                  </div>
                  {ver.warnings?.map((w) => <p key={w} className="warning small">⚠ {w}</p>)}
                  <p className="quote">“{ver.ref_text}”</p>
                  <audio controls preload="none" src={api.refUrl(v.name, ver.version)} />
                  {ver.notes && <p className="small muted">{ver.notes}</p>}
                </div>
              ))}
            </li>
          ))}
        </ul>
      </section>
      <CreateVoice />
    </div>
  );
}

function CreateVoice() {
  const qc = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const create = useMutation({ mutationFn: api.createVoice, onSuccess: (j) => setJobId(j.id) });
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => {
      const d = q.state.data;
      if (d && isDone(d.status)) {
        void qc.invalidateQueries({ queryKey: ["voices"] });
        return false;
      }
      return 1000;
    },
  });

  return (
    <section className="panel">
      <h2>New voice</h2>
      <p className="small muted">
        Use 8–15 s of one speaker, clean, in the exact delivery you want everywhere (max 30 s; longer clips make
        pitch unstable). The clip's pace and emotion carry into every generation. Saving the same name again
        creates a new version.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          create.mutate(new FormData(e.currentTarget));
        }}
      >
        <label>
          <span className="label">Name</span>
          <input name="name" required pattern="[a-z0-9][a-z0-9_\-]{0,47}" placeholder="narrator" />
        </label>
        <label>
          <span className="label">Reference audio</span>
          <input name="file" type="file" accept="audio/*" required />
        </label>
        <label>
          <span className="label">Exact transcript of the clip</span>
          <textarea name="ref_text" rows={3} required />
        </label>
        <div className="row">
          <label className="grow">
            <span className="label">Language</span>
            <select name="language" defaultValue="English">
              {LANGUAGES.map((l) => <option key={l}>{l}</option>)}
            </select>
          </label>
        </div>
        <label>
          <span className="label">Notes</span>
          <input name="notes" placeholder="e.g. calm documentary read" />
        </label>
        <button className="primary" disabled={create.isPending || (!!job.data && !isDone(job.data.status))}>
          Create voice
        </button>
      </form>
      {create.error && <p className="error">{String(create.error.message)}</p>}
      {job.data && (
        <p className={job.data.status === "failed" ? "error" : "small muted"}>
          {job.data.status === "failed" ? `Failed: ${job.data.error}` :
            isDone(job.data.status) ? `Created ${(job.data.result as unknown as { voice: string }).voice}` :
              "Building voice (loads the model if needed)…"}
        </p>
      )}
    </section>
  );
}
