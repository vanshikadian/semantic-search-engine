"use client";

import { useState, useTransition } from "react";

import { ResultCard } from "../components/ResultCard";
import { SearchBar } from "../components/SearchBar";

type SearchResult = {
  id: number;
  text: string;
  score: number;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [isPending, startTransition] = useTransition();

  const runSearch = () => {
    const trimmed = query.trim();
    if (!trimmed) {
      setError("Enter a search query to see ranked semantic matches.");
      return;
    }

    setError(null);
    setHasSearched(true);

    startTransition(async () => {
      try {
        const response = await fetch(`${API_URL}/search`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ query: trimmed, k: 10 })
        });

        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as
            | { detail?: string }
            | null;
          throw new Error(payload?.detail ?? "Search request failed.");
        }

        const payload = (await response.json()) as { results: SearchResult[] };
        setResults(payload.results);
      } catch (requestError) {
        const message =
          requestError instanceof Error
            ? requestError.message
            : "Something went wrong while searching.";
        setResults([]);
        setError(message);
      }
    });
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-6 py-10 md:px-10">
      <section className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-[radial-gradient(circle_at_top,rgba(119,214,255,0.16),transparent_35%),rgba(255,255,255,0.04)] px-6 py-8 shadow-panel backdrop-blur-xl md:px-10 md:py-12">
        <div className="absolute inset-0 animate-pulsegrid bg-[linear-gradient(120deg,transparent,rgba(119,214,255,0.05),transparent)]" />
        <div className="relative">
          <p className="mb-3 text-sm uppercase tracking-[0.32em] text-cyan-200/80">
            Embedding-Powered Retrieval
          </p>
          <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-white md:text-6xl">
            Semantic Search Engine
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-200 md:text-lg">
            Query 100K MS MARCO passages with transformer embeddings and FAISS
            exact vector search. Results are ranked by meaning, not keyword
            overlap.
          </p>

          <div className="mt-8 grid gap-3 text-sm text-slate-200 md:grid-cols-3">
            <div className="rounded-2xl border border-white/10 bg-slate-950/20 p-4">
              <div className="text-2xl font-semibold text-white">100K</div>
              <div className="mt-1 text-slate-300">passages indexed</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-950/20 p-4">
              <div className="text-2xl font-semibold text-white">FAISS</div>
              <div className="mt-1 text-slate-300">IndexFlatIP exact search</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-slate-950/20 p-4">
              <div className="text-2xl font-semibold text-white">MiniLM</div>
              <div className="mt-1 text-slate-300">all-MiniLM-L6-v2</div>
            </div>
          </div>
        </div>
      </section>

      <section className="mt-8">
        <SearchBar
          query={query}
          loading={isPending}
          onQueryChange={setQuery}
          onSubmit={runSearch}
        />
        <div className="mt-4 flex flex-wrap gap-3 text-sm text-slate-300">
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5">
            Plain-English queries
          </span>
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5">
            Cosine similarity ranking
          </span>
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5">
            Top-10 retrieval
          </span>
        </div>
      </section>

      <section className="mt-8 flex-1">
        {error ? (
          <div className="rounded-3xl border border-red-400/20 bg-red-400/10 p-5 text-red-100">
            {error}
          </div>
        ) : null}

        {!error && !hasSearched ? (
          <div className="rounded-3xl border border-white/10 bg-white/5 p-8 text-slate-300 shadow-panel backdrop-blur-sm">
            Start with a natural-language question and the engine will surface
            semantically related passages ranked by embedding similarity.
          </div>
        ) : null}

        {!error && hasSearched && results.length === 0 && !isPending ? (
          <div className="rounded-3xl border border-white/10 bg-white/5 p-8 text-slate-300 shadow-panel backdrop-blur-sm">
            No results came back for that query.
          </div>
        ) : null}

        <div className="grid gap-4">
          {results.map((result, index) => (
            <ResultCard
              key={`${result.id}-${index}`}
              rank={index + 1}
              text={result.text}
              score={result.score}
            />
          ))}
        </div>
      </section>

      <footer className="mt-10 flex flex-col gap-3 border-t border-white/10 pt-6 text-sm text-slate-400 md:flex-row md:items-center md:justify-between">
        <span>Semantic retrieval over MS MARCO with transformer embeddings + FAISS.</span>
        <div className="flex gap-4">
          <a href="https://github.com" target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a href="/eval-results">
            Eval results
          </a>
        </div>
      </footer>
    </main>
  );
}
