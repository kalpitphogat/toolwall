# Design notes

Why toolwall is built the way it is. Each section states the decision, the alternative,
and what was given up.

## 1. Threat model

An agent calls tools (fetch a page, read an inbox, search, open a file). Whatever the tool
returns goes straight into the model's context. An attacker who controls *any* of that
content (a web page, an email someone sent you, a README in a repo) can write instructions
there. This is **indirect prompt injection**. The user never typed the attack, so input
filters on the user's prompt never see it.

toolwall sits on the only channel all of that content passes through: the tool-result path
between MCP servers and the agent.

Out of scope: attacks the user types themselves (jailbreaking their own assistant),
attacks hidden in images, and an attacker who can change the gateway or its config.

## 2. Gateway, not SDK patching

**Decision:** toolwall is an MCP server to the agent and an MCP client to each real server.
Tools are re-exported as `<upstream>__<tool>`.

**Alternative:** monkeypatch the OpenAI/Anthropic/LangChain client so every response passes
through a hook.

**Why:** a proxy works with any MCP host (desktop apps, IDE agents, custom agents) without
touching the host's code, and it can't be bypassed by a code path the patch missed.
Upstream servers run as child processes of the gateway, so the agent is only ever
configured with one server.

**Cost:** tool output schemas are dropped. The gateway may rewrite or withhold a result,
and a client validating against the original schema would reject it. Hosts that rely on
structured output lose it for proxied tools.

## 3. Three actions instead of one threshold

| verdict | action |
|---|---|
| allow | pass through untouched |
| warn  | pass through inside `<<<UNTRUSTED_TOOL_OUTPUT … >>>` markers, with a notice telling the model to use the content as data and not follow instructions in it ("spotlighting") |
| block | withhold, return an error with the score and audit event id |

A single block threshold forces a bad trade-off: set it low and you destroy useful tool
output; set it high and borderline attacks pass untouched. The warn band keeps the
content usable while removing its authority. Thresholds are named by the false-positive
rate they were fitted for on validation (`fpr_0.05` warns, `fpr_0.01` blocks), so the
config states the trade-off directly instead of an opaque probability.

`monitor` mode logs every verdict and never alters output. That's the safe way to roll it
out: measure the false-positive rate on your own traffic first.

## 4. Data: why the splits are built the way they are

Public injection datasets are small, overlapping and full of near-copies. A model
evaluated on a random split of them mostly measures memorisation, and so does a public
model evaluated on its own training set. Four rules guard against that:

1. **Group-level splits.** The unit that goes to train/val/test is a *group*, not a row.
   For real attack emails (LLMail-Inject) the group is the attacker team, so the test set
   contains only attackers the model never saw. For short prompts the group is a
   near-duplicate cluster.
2. **Near-duplicate clustering** with MinHash/LSH over word 3-gram shingles (Jaccard ≥ 0.8,
   verified exactly after LSH). Clusters whose members carry different labels are dropped
   as ambiguous.
3. **Held-out-first purging.** The two out-of-distribution sets are loaded first. Any
   prompt that is a near-copy of one of them is removed *before* synthesis, because once a
   payload is pasted into a long carrier document, whole-text similarity no longer finds
   it. Afterwards val and train are purged against test and OOD.
4. **Independent audit.** A second pass re-checks every held-out split against train, on
   whole texts and on inserted payloads, and writes the counts to `manifest.json`.
   Every count is expected to be zero.

### Synthesis without shortcuts

Real indirect-injection data is scarce, so most training examples are synthesised: a
payload inserted into a benign carrier document (Wikipedia-style paragraphs, news
articles) rendered as a web page, email, search-result JSON, chat log, ticket, markdown
or plain text, and hidden as a paragraph, an HTML comment, a footer, a quote, a bracketed
note or inline.

The obvious failure is a classifier that learns "HTML comment ⇒ attack" or
"email ⇒ attack" (every LLMail positive is an email). So **every wrapper and every
concealment style is used for negatives too**, with benign inserts that are often
themselves imperatives ("Write a short summary of…", "Please send the report by Friday").
An equal number of benign emails is generated per split to cancel out the all-email
LLMail positives. What's left to learn is *what the inserted text asks for and whom it
addresses*.

## 5. Long documents: windows, not truncation

Encoders see 512 tokens, and tool outputs are often longer, with the injection anywhere,
often at the end. Truncation silently drops the tail. Both training and inference split
text into overlapping 512-token windows (stride 128). The document score is the max over
windows, and that window's character range is returned as evidence.

During training, payload spans are known for synthetic data, so only windows that overlap
the payload are positive. The other windows of the same document become extra benign
context. Real attack emails without a known span are kept only if they fit in one
window. Multi-window ones are dropped (their count is logged) rather than mislabelled.

## 6. Calibration and thresholds

Raw classifier probabilities are over-confident and change from run to run. After
training, a single temperature is fitted on validation (NLL, 1-D search), and thresholds
are chosen on validation negatives for target false-positive rates. The test and OOD sets
are never used for any of this, so the reported test FPR is an honest estimate.

## 7. One scoring path

The evaluation harness, the HTTP API and the gateway all call `Scorer.scan_many`. Benchmark
numbers therefore describe exactly the code that runs in production: same windowing,
same calibration, same thresholds. A separate "benchmark scorer" is how projects end up
reporting numbers their product does not achieve.

## 8. Audit log

SQLite (WAL) with one row per scanned tool result: verdict, score, whether it was
enforced, latency, SHA-256 of the content and arguments, and an excerpt *only* for
flagged results (configurable off). The log is for answering "what did the agent read,
and what did we do about it", without becoming a second copy of everything the agent saw.

## 9. Known limitations

- English-centric data. Non-English injections are under-represented.
- Static detector. An attacker with query access can search for phrasings that score
  low. The warn band and audit log reduce the damage, but don't prevent it.
- Detection is not proof of safety. A clean verdict does not mean the content is
  harmless, only that it doesn't look like an instruction aimed at the agent.
- Non-text content (images, binary resources) passes through unscanned.
- Scoring adds latency to every tool call (see the report for measured numbers).
