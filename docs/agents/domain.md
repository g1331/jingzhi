# Domain docs

This repository uses a single domain context.

Before exploring or changing the system, read:

- `CONTEXT.md`, when present.
- Relevant decisions under `docs/adr/`, when present.

These files are created lazily as terminology and architectural decisions are
resolved. Their absence is not an error.

Use terminology defined in `CONTEXT.md` consistently in issues, specifications,
tests, and code. If proposed work conflicts with an ADR, identify the conflict
explicitly instead of silently overriding the decision.

Expected layout:

```text
/
├── CONTEXT.md
├── docs/adr/
└── src/
```
