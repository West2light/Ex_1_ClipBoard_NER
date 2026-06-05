# Project Agent Memory

## Why this folder exists

Codex native Memories are enabled globally on this machine and stored under:

```text
~/.codex/memories/
```

Native memories can carry useful context between eligible threads, but current
Codex documentation does not define a separate durable memory namespace for
each custom subagent name.

This folder adds explicit project-scoped, per-agent memory files:

```text
phone-reviewer.md
ner-evaluator.md
api-architect.md
api-worker.md
```

## Behavior

Each custom agent is instructed to:

1. read `AGENTS.md`
2. read its own memory file
3. update only its own memory file

This provides durable and reviewable project knowledge without depending only
on native memory extraction.

## What to record

Record:

- confirmed project facts
- approved technical decisions
- recurring pitfalls
- reusable verification commands
- stable ownership boundaries

Do not record:

- API keys or secrets
- raw customer text
- real phone numbers or PII
- transient command logs
- unverified speculation
- large copied outputs

## Precedence

When instructions conflict:

```text
latest user request
-> AGENTS.md
-> approved project plans
-> per-agent memory
-> native recalled memory
```

## Coordination

Review agents may write only their own memory file. They must not edit
application code or another agent's memory.

Use only one implementation worker at a time to reduce write conflicts.

