---
name: planning
description: Plans next steps for the user based on their objective and current codebase state. Produces incremental, learning-friendly steps so the user can build working features step-by-step. Use when the user asks for a plan, next steps, roadmap, or when they have an objective they want to achieve in the codebase.
---

# Planning

## Purpose

Given the user's objective and the current codebase, produce a concrete plan of next steps. Each step should:
- Be small enough to complete and verify before moving on
- Build on previous steps
- Result in working, testable code
- Support incremental learning (user learns by doing, not by reading large blocks)

## Planning Workflow

### 1. Gather Context

Before planning:
- **User objective**: What does the user want to achieve? (feature, fix, understanding)
- **Codebase state**: What exists? What's implemented vs. planned?
- **Project phases**: If in Agent_ng, reference `.github/copilot-instructions.md` for canonical phases

### 2. Map Objective to Current State

- Identify what's already done vs. what's missing
- Find the smallest viable next step toward the objective
- Respect dependencies (e.g., don't plan RAG retrieval before embeddings work)

### 3. Produce the Plan

Output format:

```markdown
## Plan: [Brief objective]

**Current state:** [1–2 sentences on what exists]
**Goal:** [What we're working toward]

### Next steps

1. **[Step name]** — [One clear deliverable]
   - What to do: [Brief instruction]
   - How to verify: [How user knows it works]
   - [Optional: small code hint or file to touch]

2. **[Step name]** — [Next deliverable]
   ...

**After each step:** Try it. If issues arise, resolve before proceeding.
```

### 4. Execute One Step at a Time

- Present only the **first step** in detail
- Provide minimal code/suggestions for that step
- Wait for user to try and confirm (or report issues)
- Only then reveal and implement the next step

## Principles

| Principle | Apply by |
|-----------|----------|
| **Incremental** | One step at a time; no large dumps |
| **Verifiable** | Each step has a clear "done" check |
| **Working** | Every step leaves the codebase in a runnable state |
| **Learning** | Small chunks; user tries, agent helps fix |

## Anti-Patterns

- ❌ Listing 10 steps and implementing all at once
- ❌ Skipping verification ("assume it works")
- ❌ Planning steps that depend on unbuilt foundations
- ❌ Large code blocks before user has tried smaller ones

## Agent_ng Phase Reference

When planning in Agent_ng, align with phases in `.github/copilot-instructions.md`:

- Phase 0: Repo & environment
- Phase 1: Core primitives (config, LLM, embeddings, vector DB)
- Phase 2: Document ingestion
- Phase 3: RAG & retrieval
- Phase 4+: Speech, tools, agents, workflows, etc.

Use phases to order steps; don't jump ahead without prerequisites.
