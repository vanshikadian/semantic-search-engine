"use client";

import { FormEvent } from "react";

type SearchBarProps = {
  query: string;
  loading: boolean;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
};

export function SearchBar({
  query,
  loading,
  onQueryChange,
  onSubmit
}: SearchBarProps) {
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit();
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-[2rem] border border-white/10 bg-white/5 p-3 shadow-panel backdrop-blur-md"
    >
      <div className="flex flex-col gap-3 md:flex-row">
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search for ideas like: how do astronauts sleep in space?"
          className="min-h-14 flex-1 rounded-2xl border border-white/10 bg-slate-950/30 px-5 text-base text-white outline-none transition focus:border-cyan-300/40 focus:bg-slate-950/50"
        />
        <button
          type="submit"
          disabled={loading}
          className="min-h-14 rounded-2xl bg-gradient-to-r from-cyan-300 via-sky-400 to-orange-300 px-6 font-semibold text-slate-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-70"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
    </form>
  );
}
