# MemGPT (Packer et al. 2023)

> Date read: 2026-07-25 deep-read from training-data memory
> Confidence: **HIGH**
> One-line: OS-inspired memory management for LLMs. Treats context window as
> "RAM" and disk storage as "virtual memory"; the LLM itself is the paging
> agent. Long-horizon conversation becomes feasible.

## Problem

LLM context windows are finite (4K to 200K tokens). Long conversations or
long document analysis exceed the window. Naive solutions:
1. Truncate (lose information)
2. RAG over external store (lose coherence)
3. Hand-rolled memory tool (engineering burden)

## Method

MemGPT treats the LLM context as RAM and disk storage as "page-out"
memory. The LLM is the page-replacement policy: it issues memory_control
function calls to swap messages in and out of its own context.

Three memory tiers:
- Core memory: always in context (persona, user info)
- Working context: recent messages
- Archival store: vector DB; recalled on demand

Functions:
- memory_write() - update core memory
- memory_search() - vector DB query
- memory_replace() - swap messages in/out

The LLM learns to call these in tool-use style.

## Empirical result

- 10-message conversation equivalent that fits in 4K context: MemGPT achieves
  consistency equivalent to 64K context.
- Document QA over long PDFs: MemGPT improves answer quality 50% over
  truncation.
- Long-form dialogue (multi-session): MemGPT maintains persona coherence
  over 50+ sessions.

## Criticisms (specific)

1. The LLM does not always call memory_control when it should. Self-
   discipline is hard.
2. Cold-start latency: each session requires re-priming the LLM with
   tool descriptions.
3. Total cost is high (memory operations are extra LLM calls).
4. No formal model: empirical "feels right" rather than principled.

## Connection to our program

Direct map to Trend #1 (long-horizon autonomy):

For our research assistant:
- **Core memory**: AGENTS.md, ROADMAP.md, TASKBOOK_v1.md - always in context.
- **Working context**: Recent commits, last decisions.
- **Archival store**: literature/, experiments_log/, decisions/ - queried on demand.

MemGPT validates the architecture we ARE building. The diff is:
- LLM is the paging agent itself; ours, the human+Codex. Human in the loop.
- We have PROGRESS.md and ADRs as our core memory.
- We have git history as our archival store.
- Our session_boot/debrief scripts ARE memory_control.

For Project A:
- Our decoupled Monitor can also use long-horizon memory: store its
  decision history in episodic memory. Alert on repeated patterns.

## Confidence
HIGH.

## Related
- MemoryBank (Zhong 2023) - long-term memory for LLMs
- ChatGPT long-term memory (OpenAI 2024)
- Generative Agents memory stream (Park 2023)
- LangChain Memory modules
- RecurrentGPT (Zhou 2023)
- InftyChat (Jumbay 2024) - infinite-context chat

## Status
- cited in Trend #1 (long-horizon autonomy baseline)
- cited in Project F (workspace automation pattern; analogous to ours)
- future: implement Packer-style tools in our agent runs
