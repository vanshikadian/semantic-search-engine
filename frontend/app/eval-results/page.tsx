import fs from "node:fs";
import path from "node:path";

type EvalPayload = {
  evaluated_queries: number;
  aggregate: Record<string, number>;
  split: string;
};

function loadEvalResults(): EvalPayload | null {
  const filePath = path.join(
    process.cwd(),
    "..",
    "backend",
    "faiss_index",
    "eval_results.json"
  );

  if (!fs.existsSync(filePath)) {
    return null;
  }

  const raw = fs.readFileSync(filePath, "utf8");
  return JSON.parse(raw) as EvalPayload;
}

export default function EvalResultsPage() {
  const data = loadEvalResults();

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-4xl flex-col px-6 py-10 md:px-10">
      <div className="rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-panel backdrop-blur-sm">
        <p className="text-sm uppercase tracking-[0.32em] text-cyan-200/80">
          Benchmark Results
        </p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white">
          Retrieval evaluation
        </h1>

        {!data ? (
          <p className="mt-6 text-base leading-7 text-slate-300">
            No evaluation artifact found yet. Run
            <code className="mx-2 rounded bg-white/10 px-2 py-1 text-sm text-white">
              python scripts/evaluate.py
            </code>
            and refresh this page.
          </p>
        ) : (
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-white/10 bg-slate-950/20 p-4">
              <div className="text-2xl font-semibold text-white">
                {data.evaluated_queries}
              </div>
              <div className="mt-1 text-slate-300">queries evaluated</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-950/20 p-4">
              <div className="text-2xl font-semibold text-white">
                {(data.aggregate["recall@5"] * 100).toFixed(1)}%
              </div>
              <div className="mt-1 text-slate-300">Recall@5</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-950/20 p-4">
              <div className="text-2xl font-semibold text-white">
                {(data.aggregate["recall@10"] * 100).toFixed(1)}%
              </div>
              <div className="mt-1 text-slate-300">Recall@10</div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
