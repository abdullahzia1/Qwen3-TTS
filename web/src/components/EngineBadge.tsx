import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type EngineState } from "../api";

const LABEL: Record<EngineState, string> = {
  stopped: "Model stopped",
  loading: "Loading model…",
  ready: "Model ready",
  busy: "Generating",
  stopping: "Stopping…",
  error: "Model error",
};

export function EngineBadge() {
  const qc = useQueryClient();
  const { data } = useQuery({ queryKey: ["engine"], queryFn: api.engine, refetchInterval: 1500 });
  const act = useMutation({
    mutationFn: (start: boolean) => (start ? api.startEngine() : api.stopEngine()),
    onSuccess: (e) => qc.setQueryData(["engine"], e),
  });
  if (!data) return <div className="engine">…</div>;

  const canStart = data.state === "stopped" || data.state === "error";
  const canStop = data.state === "ready" || (data.state === "busy" && !data.stop_pending);
  const idleLeft = data.idle_timeout_s ? Math.max(0, Math.round((data.idle_timeout_s - data.idle_s) / 60)) : null;

  let detail = "";
  if (data.state === "ready" && idleLeft !== null) detail = `auto-stop in ${idleLeft} min`;
  if (data.state === "busy" && data.stop_pending) detail = "stops after current task";
  if (data.queue_depth) detail = `${data.queue_depth} queued${detail ? " · " + detail : ""}`;

  return (
    <div className="engine" title={data.error ?? data.model?.name_or_path ?? ""}>
      <span className={`dot ${data.state}`} />
      <span>{LABEL[data.state]}</span>
      {detail && <span className="muted small">{detail}</span>}
      {canStart && (
        <button disabled={act.isPending} onClick={() => act.mutate(true)}>
          Start model
        </button>
      )}
      {canStop && (
        <button className="ghost" disabled={act.isPending} onClick={() => act.mutate(false)}>
          Stop model
        </button>
      )}
    </div>
  );
}
