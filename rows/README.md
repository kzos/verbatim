# `rows/` — published measurement rows, as data

Empty on purpose. **No number that the harness did not produce**, so until the harness produces one
there is nothing here, and the README's demo blocks keep their angle brackets.

Layout: `rows/<die>/<checkpoint>/<chunk_mode>/<arm>-<date>-<handle>/` — one directory per run, holding
the results JSON (schema `vb-results/1`) and its attachments. Produced by `verbatim-bench run` plus
`verbatim-bench row`, never hand-written, and validated in CI with no human in the loop.

A row is this project's contribution unit. See `CONTRIBUTING.md` §1 and
the results schema and the `verbatim-bench verify` command.
