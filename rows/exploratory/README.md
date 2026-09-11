# `rows/exploratory/` — runs that are NOT rows

A row under `rows/` is a measurement the day-21, day-45, day-60 and day-90 gates are read against. It
must be canonical, comparable and reproducible. Everything in **this** directory failed at least one of
those and is kept because the failure is informative.

Nothing here may be cited as a benchmark result, quoted in a comparison, or used to argue a gate either
way. Each file says in its own record why it is not a row.

| file | why it is not a row |
|---|---|
| `ladder-b300-bf16-graphed-2026-09-11.json` | Same server, same corpus, same canonical window, on the graph path. It reports `s: 0`, which describes the search and not the server: one seed was non-monotonic, failing at 6 streams and passing at 7, and a ladder search assumes monotonicity. Its p95 values fall in two clusters, roughly 250-282 ms and 321 ms with nothing between, so a rung passes or fails by which mode it lands in. Evidence section 24. |
| `ladder-a6000-bf16-eager-2026-09-11.json` | The window is canonical and the run is clean, but the three seeds located the boundary at N=4, 6 and 11. The server sits on its 310 ms latency budget at every concurrency tested, so pass/fail is decided by about 20 ms of noise rather than by load. The reported `s: 4` is the low end of a spread, not a capacity. Evidence section 23. |
