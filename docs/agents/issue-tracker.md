# Issue tracker: GitHub

Issues and PRDs for this repository live in GitHub Issues. Use the `gh` CLI
and infer the repository from its Git remote.

## Operations

- Create, read, list, comment on, label, and close issues with `gh issue`.
- Publishing to the issue tracker means creating a GitHub issue.
- Fetching a ticket means reading the issue and its comments and labels.
- Pull requests are not treated as incoming requests for triage.
- Use GitHub sub-issues and native issue dependencies where available.
- If dependencies are unavailable, record `Blocked by: #<number>` in the issue.
- A ticket is ready only when all blockers are closed.
