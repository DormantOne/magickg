Each layer is a response to a specific failure mode observed in earlier
versions. The rationale for each layer is below.

### Layer 1: Heat Map (perception)

**Problem it solves:** LLMs cannot reliably read spatial ASCII. They process
board strings as token sequences, losing the 2D structure. Small models
hallucinate "the white group at the bottom-left" when no such group exists.
All downstream reasoning is polluted by faulty perception.

**What it does:** Six pure-Python heat functions run on the board before any
LLM is invoked. They compute, for every legal empty point, how much a move
there would matter:

- `heat_capture` — points that capture enemy groups
- `heat_save` — points that save own groups from atari
- `heat_liberty_reduce` — points that reduce enemy liberties
- `heat_connect_or_cut` — connection/cut points
- `heat_territory` — contested territory points
- `heat_edge_structure` — star points (decays as game fills)

The values are summed into a 9×9 heat grid. Top-K hottest points (default 5)
are selected with non-max suppression. Each has a dominant component label —
the one reason it's hot.

Then one LLM call (the *heat analysis*) gets the board, heat map, and anchors,
and produces a 3-4 sentence position read. The resulting read is stored as a
`HEAT_READ` node in the KG.

**Why this matters:** The brain no longer reasons from raw ASCII. It reasons
from an accurate, computable, interpretable description of where the action
is. And because heat reads are stored, the KG accumulates a traceable
perceptual memory across games.

### Layer 2: Tools (self-modifying analysis functions)

**Problem it solves:** Even with perception, there are specific numeric
questions the LLM is bad at — "how many liberties does the group at (3,4)
have?" If the LLM tries to count them itself, it's often wrong.

**What it does:** Three seeded tools compute the numbers reliably:
- `count_liberties` — group size and liberty count
- `eval_territory` — stone counts and influence
- `find_atari` — groups with one liberty

Each tool has a **replay value** updated from outcomes. Tools that produce
positive score deltas gain replay value; tools that correlate with bad moves
lose it. Dead tools can be archived. The brain can create new tools via
`[SAVE_TOOL:...]` syntax during a move if it needs analysis it doesn't have.

In practice the brain mostly doesn't create new tools. That itself is a
finding — small models struggle to write useful analysis code under pressure.

### Layer 3: Lens Selection

**Problem it solves:** The LLM needs context about its own goals this turn
(attack? defense? territory?) to retrieve the right knowledge.

**What it does:** One LLM call picks from a fixed list of lenses
`[defense, attack, territory, endgame, tactics]`. The lens weights subsequent
knowledge retrieval — cognitive nodes with router scores matching the lens
rank higher.

### Layer 4: Knowledge Selection (funneled + critic-weighted)

**Problem it solves:** The KG accumulates many cognitive nodes; retrieving
them all poisons the move prompt. But picking the "right" N is hard, and
past versions just used relevance, which doesn't account for whether the
knowledge is *true*.

**What it does:** A funnel — when there are few cognitive nodes, a single
pass selects; when there are more, progressive LLM passes narrow down. Final
ranking combines:
- Relevance score (learned over time)
- Router score for the active lens
- **Critic score** — how often the node has passed two-board fit tests

The formula: `relevance × (0.5 + critic_score) + router × 0.5`. A node the
brain trusts (high relevance) but the critic doesn't (low critic score) is
down-weighted at retrieval. This is where the critic earns its keep during
normal play.

### Layer 5: Graph Expansion

**Problem it solves:** A selected insight may link to important related
knowledge via KG edges.

**What it does:** For each selected cognitive node, follow its edges to
connected nodes, add top connections to the retrieved set. Capped at a
small number to avoid bloat.

### Layer 6: Working Strategies (live plans with progress tracking)

**Problem it solves:** The brain generates knowledge but has no *temporal*
accountability. An insight like "build walls" can't be tested. A plan that
the brain declares, executes, and tracks *can* be tested — against the
actual outcome of the moves that followed.

**What it does:** On each move, the brain either continues an active
working strategy or declares a new one with `[STRATEGY:N] plan text`, where
N is a horizon (3-15 moves).

The WS node snapshots:
- The plan text
- The horizon
- The board state when declared (score, group counts, stone counts)
- The moves played under the plan

On each subsequent move, the brain sees its plan plus *live progress*: score
when declared, score now, margin delta. "The plan is working" / "failing" /
"even" is in the prompt. This is the feedback loop that was missing from
earlier versions — the brain can tell, mid-plan, whether its current intent
is working.

When the WS reaches its horizon (or the brain declares a new one), it
**resolves** as succeeded / failed / neutral based on margin delta thresholds
(±3). Resolved WS with meaningful outcomes get **distilled** into permanent
`INSIGHT` or `STRATEGY` nodes with concrete evidence attached: "Plan 'capture
the black stone at (2,1)' worked: executed over 3 moves, margin improved by
+13.0. Moves: (2,1), (3,0), (4,2)".

Distilled nodes are the richest cognitive content the brain can accumulate.
They carry built-in evidence. Their critic history is inherited from the
WS they came from.

### Layer 7: Move Generation

The LLM gets the board, the heat read, the tool outputs, the retrieved
knowledge, the active working strategy with live progress, and the identity
and drives. It outputs analysis, optionally a new `[STRATEGY:N]`
declaration, optionally a new `[SAVE_TOOL]`, and the move.

---

## The Critic (fresh-eyes two-board fit test)

**Problem it solves:** In earlier versions, cognitive nodes entered the KG
at high confidence without any check. Mid-game synthesis would generate
plausible-sounding spatial claims and store them at relevance 1.0. They'd
be retrieved on the next move, justifying moves that lost. Over games, the
KG became a hallucination echo chamber.

**What it does:** A separate LLM call with no conversation history tests
whether a claim actually describes a board. Given:
- A claim (cognitive node content)
- Two boards in randomized order (A and B)
- Prompt: "Which board does this claim better describe? Think step by step."

The critic returns one of:
- `VERDICT: A` — claim better fits board A
- `VERDICT: B` — claim better fits board B
- `VERDICT: EITHER` — claim too vague to distinguish

The caller knows which was the "home" board (the board the claim was derived
from). If the critic picks the home board → **match**. Picks the other →
**miss**. Says either → **vague**.

Each match counts as a pass. Miss and vague count as fails. A node's
**critic score** is the Laplace-smoothed pass rate over recent tests (capped
at 10 in history). The score is used in retrieval ranking — unreliable
claims still exist, but are down-weighted.

**When the critic runs:**
- **Birth test** — every new cognitive node (from review OR mid-game
  synthesis) gets one critic call before it's fully admitted. Home board is
  the most recent conversation's board; comparison is a random other-game
  board. Failed/vague claims have relevance multiplied by 0.4 at birth.
- **Dream sample** — per dream cycle (post-game), 3 sampled existing nodes
  get re-tested. Nodes with critic_score < 0.35 over 3+ tests get their
  relevance hammered by ×0.5.

**Why this mechanism matters:** Single tests have 50% luck baseline. Multiple
tests over a node's lifetime converge on truth. Bad nodes accumulate failures
they can't recover from; good nodes validate consistently. The critic
doesn't censor — it down-weights in proportion to empirical track record.

Critic telemetry is surfaced in the biopsy: lifetime calls, match/miss/vague
distribution, latency, by-source counts, recent verdicts.

---

## The Knowledge Graph

Nodes have a type (insight, strategy, theory, goal, working_strategy,
conversation, heat_read, tool, identity, drive, meta) and carry:
- `content` — text
- `relevance_score` — learned weight, affects retrieval
- `critic_score` — how often the node has passed fit tests
- `critic_history` — recent pass/fail list
- `replay_value` — for tools, outcome-linked usefulness
- `router` — per-lens affinity weights
- `meta` — type-specific data (plan state, heat anchors, etc.)

Edges carry channels (labeled affinities) and confidence. They link
cognitive nodes to the conversations they came from, working strategies to
the plays they produced, heat reads to the conversations they read, distilled
insights to the working strategies that produced them, and related cognitive
nodes to each other.

Knowledge **decays** over time when unused. It **archives** below a relevance
threshold. Tools are **protected** from decay — they live or die by replay
value alone.

---

## Seeded vs Self-Populated Content

A deliberate experimental choice: **the brain starts with no seeded Go
knowledge.** Identity and drives are seeded (who the brain is, what motivates
it) but no Go strategy, no opening principles, no priorities.

Earlier versions had three seeded goals — "secure corners," "count
liberties," "build solid groups." These confounded the experiment: you
couldn't tell if the KG was learning or if the seeds were driving play. And
several of them were actively harmful against a bare LLM opponent on 9×9,
where passive territorial play loses to tactical aggression.

With no seeds, **every piece of Go knowledge in the KG came from lived
experience**, filtered through the critic. When the brain wins, it wins with
knowledge it earned.

The brain can populate its own goals via `[GOAL]` tags in post-game review or
synthesis. In runs we've seen, it has declared goals like "Preserve at least
one free eye in every major territory" — a Go concept, articulated by the
brain itself, after games where eye-preservation mattered.

---

## The Experimental Loop

Two-phase rounds:

1. **Train** — brain (White) vs bare LLM (Black). Full pipeline. KG
   mutates. Post-game: review (3 LLM passes), dream cycle (decay, routing,
   critic sampling, tool review), working-strategy distillation. Snapshot
   brain state.

2. **Probe** — frozen brain_gN vs bare LLM. No DKG mutation. This is the
   honest test: can the accumulated knowledge, as of a specific moment,
   produce good play against the same opponent?

Between rounds, the brain persists to `data/brain.json`. The learning
survives restarts.

---

## The Biopsy

`/api/game/biopsy` returns a structured dump of everything: current game
metrics, pipeline breakdown, tool effectiveness, board state, training and
probe records, round history, active working strategy with live progress,
recent resolved plans, cognitive nodes with critic scores and histories,
critic telemetry (lifetime and this-game), latest heat analysis with anchors
and decomposition, KG formation stats (nodes by type, edges, archival),
lens distribution, tool calls, snapshot list.

The biopsy is designed to be readable by other agents — everything has a
decision trail. If a cognitive node has critic_score 0.67, you can see which
tests it passed and which it failed. If a plan is listed as succeeded with
Δ+13, you can find the move sequence. If the heat analysis chose (3,0) as
the critical point, you can see the heat components that made it hot.

This transparency is a deliberate architectural goal. The KG isn't a
black box you hope is learning; it's an audit trail you can inspect.

---

## Running

Requires [Ollama](https://ollama.com) running locally. Default model is
`gpt-oss:20b` but you can change it in `ollama_client.py`. Larger models
should work better — the 20b model is at the low end of what can handle this
pipeline.

```bash
pip install -r requirements.txt  # or just: flask
ollama serve  # in another terminal
python game_app.py  # starts on port 5028
open http://localhost:5028
```

Click "Auto ▶" to start the training/probe loop. Click "bx" to see biopsies
as the game runs.

### Configuration (`config.json`)

Key knobs:
- `board_size` — default 9
- `komi` — default 5.5 (White's compensation for Black going first)
- `move_cap` — default 120
- `opponent_type` — `bare_llm` (default) / `heuristic` / `random`
- `enable_critic` — default true
- `enable_working_strategy` — default true
- `enable_heat_analysis` — default true
- Various timeouts

---

## What This Experiment Can and Cannot Show

**Can show:** Whether a KG with perception, critic, plans, and self-generated
goals lets a small LLM play Go better than the same LLM bare.

**Cannot show:** Whether this scales to harder tasks, larger models, or
different domains. Go has a clean outcome signal (margin) which makes
learning loops tractable. Many domains don't.

**Also worth being honest about:** The brain's "Go knowledge" is shaped by
the opponent. Against a bare LLM of the same size, one set of patterns
works. Against a stronger opponent, different patterns would emerge. The
architecture is a substrate; what it learns depends entirely on what it
plays against.

---

## Observed Results (v13, as of this writing)

First round of v13 with no seeded goals:

- Training G1: **White 141.5 — Black 4.0 (margin +137.5)**. Decisive win.
- Five working strategies distilled into lasting strategy nodes, each
  carrying move-sequence evidence.
- One self-generated goal: "Preserve at least one free eye in every major
  territory."
- Critic: 25 verdicts across synthesis-birth and dream-sample, with
  meaningful distribution (32% match, 16% miss, 52% vague — a healthy mix).
- Dream cycle hammered 2 existing cognitive nodes whose critic scores
  stayed low across multiple tests.

Prior versions (v11, v12) lost by -80 to -100 margins consistently. The
difference is attributable to the combination of perception (heat map),
critic wiring at all synthesis paths, and removal of seeded goals that
were biasing retrieval.

---

## Limitations and Open Questions

- Mid-game synthesis is still an expensive operation. A dense run with
  every synthesis call generating critic birth tests is slow.
- Dream sampling only touches 3 nodes per game. Large KGs will have
  untested high-relevance nodes for long periods.
- The heat map's component weights are hand-tuned. Systematic tuning
  would likely improve anchor selection.
- The working-strategy margin-delta thresholds (±3 for succeeded/failed)
  are heuristic. Most plans currently resolve as neutral.
- The architecture has not been tested across opponents of different
  strengths. A stronger opponent might produce very different KG content.
- There is no analog retrieval over heat reads yet. When the brain faces
  a board similar to one it has seen, it does not explicitly retrieve
  the prior heat read. This is a clear next layer.

---

## Files

- `game_app.py` — main loop, pipeline, critic, heat map, working strategies,
  review, synthesis, biopsy, Flask endpoints
- `dkg_engine.py` — Node, Edge, DKG class, store/retrieve/decay/archive logic
- `go_engine.py` — Go rules, group/liberty logic, scoring
- `ollama_client.py` — LLM call wrapper
- `selfmod_tools.py` — tool execution
- `templates/game.html` — UI
- `config.json` — runtime config
- `data/` — runtime state (brain.json, snapshots, scoreboards)

---