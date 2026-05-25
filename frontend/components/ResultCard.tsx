type ResultCardProps = {
  rank: number;
  text: string;
  score: number;
};

export function ResultCard({ rank, text, score }: ResultCardProps) {
  const scorePercent = `${(score * 100).toFixed(1)}%`;

  return (
    <article className="animate-fade-up rounded-3xl border border-white/10 bg-white/5 p-5 shadow-panel backdrop-blur-sm">
      <div className="mb-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-cyan-300/20 bg-cyan-300/10 text-sm font-semibold text-cyan-200">
            {rank}
          </span>
          <span className="text-xs uppercase tracking-[0.24em] text-slate-300/70">
            Ranked result
          </span>
        </div>
        <span className="rounded-full border border-orange-300/20 bg-orange-300/10 px-3 py-1 text-sm font-medium text-orange-100">
          {scorePercent}
        </span>
      </div>
      <p className="text-base leading-7 text-slate-100">{text}</p>
    </article>
  );
}
