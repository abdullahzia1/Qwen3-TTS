import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { fmt } from "../components/Waveform";

export function History({ onOpen }: { onOpen: (id: string) => void }) {
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 3000 });
  return (
    <section className="panel">
      <h2>History</h2>
      {jobs.data?.length === 0 && <p className="muted">Nothing generated yet.</p>}
      <table className="history">
        <thead>
          <tr>
            <th>When</th>
            <th>Voice</th>
            <th>Script</th>
            <th>Length</th>
            <th>Status</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.data?.map((j) => (
            <tr key={j.id}>
              <td className="small mono">{new Date(j.created_at).toLocaleString()}</td>
              <td className="mono">{j.request.voice}</td>
              <td className="excerpt">{j.request.text.slice(0, 90)}{j.request.text.length > 90 ? "…" : ""}</td>
              <td className="mono">{j.result ? fmt(j.result.meta.duration_s) : "—"}</td>
              <td><span className={`badge ${j.status}`}>{j.status}</span></td>
              <td><button className="ghost small" onClick={() => onOpen(j.id)}>Open</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
