import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { EngineBadge } from "./components/EngineBadge";
import { History } from "./pages/History";
import { Studio } from "./pages/Studio";
import { Voices } from "./pages/Voices";
import "./styles.css";

type Tab = "studio" | "voices" | "history";

function App() {
  const [tab, setTab] = useState<Tab>("studio");
  const [jobId, setJobId] = useState<string | null>(null);
  return (
    <>
      <header>
        <span className="brand">voicekit</span>
        <nav>
          {(["studio", "voices", "history"] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? "tab active" : "tab"} onClick={() => setTab(t)}>
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <EngineBadge />
      </header>
      <main>
        {tab === "studio" && <Studio jobId={jobId} onJob={setJobId} />}
        {tab === "voices" && <Voices />}
        {tab === "history" && <History onOpen={(id) => { setJobId(id); setTab("studio"); }} />}
      </main>
    </>
  );
}

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={qc}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
