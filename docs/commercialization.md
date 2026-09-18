# Commercialization Plan (lightweight)

> **Discipline:** this document separates **technical possibility** (what the
> system can do), **market hypothesis** (what we believe but have not proven), and
> **validated revenue** (what customers have actually paid for). As of this
> writing there is **no validated revenue and no market validation** — every
> demand claim below is an explicit hypothesis. The specialized model's quality is
> also still **UNPROVEN** (see the model status), which gates any paid offering.

## Current honest status

| Layer | Status |
|---|---|
| Technical possibility | CLI + web + API over a shared engine, deployable via compose. **Real.** |
| Differentiated model | A cyber-specialized model that beats base gemma3:4b. **Not yet demonstrated.** |
| Market hypothesis | Documented below. **Untested.** |
| Validated revenue | **None.** |

The product cannot honestly be sold as "a better cybersecurity model" until
exp-002 (or a successor) clears its pre-registered evaluation gate. Until then the
sellable asset is the *packaging* (a private, safe, self-hostable cyber assistant),
not model superiority.

## Target customer (hypothesis)

Primary: **individual security practitioners and small blue teams / MSSPs** who
want a *private, self-hostable* assistant for defensive analysis, ATT&CK mapping,
and study — and who are uncomfortable pasting internal detail into a public
chatbot. Secondary: **security learners / CTF players** who want a local tutor.

## Core problem (hypothesis)

General chatbots are (a) not private/self-hostable, (b) not tuned for
evidence-grounded, hallucination-averse security reasoning, and (c) happy to
fabricate CVE/ATT&CK specifics. The wedge is **privacy + calibrated uncertainty**,
not raw capability against a frontier model.

## Product value

- Runs locally / self-hosted — data never leaves the operator's environment.
- Safety-forward defaults: flags insufficient evidence, resists fabricated premises.
- Reproducible, versioned model with an auditable promotion trail.

## Free vs paid (proposed MVP split)

| Tier | Price (hypothesis) | Contents |
|---|---|---|
| **Free / OSS** | $0 | Local CLI + self-host the API/web against your own Ollama |
| **Pro (hosted)** | ~$10–20 / mo | Managed hosted assistant, no self-host ops, higher rate limits |
| **API** | usage-based | Per-request/token API for integration into other tools |
| **Team / Enterprise** | custom | SSO, audit export, self-host license + support |

Keep the free local tier genuinely useful — it is the acquisition channel and the
credibility proof for the paid, hosted convenience.

## Unit economics considerations (not yet measured)

- **Inference cost:** gemma3:4b is cheap (CPU-servable via Ollama). A hosted Pro
  tier's main cost is the always-on host, not per-token GPU. This is a deliberate
  advantage of a small specialized model over reselling frontier API calls.
- **Hosted infra:** one small VM can serve early Pro users; costs scale with
  concurrency, not signups (stateless per request today).
- **Gross margin** depends entirely on hosted concurrency vs subscription price —
  **must be measured with real load before pricing is committed.**

## Customer-acquisition hypothesis

OSS/free local tool → GitHub + security-community visibility → a share convert to
hosted Pro for convenience. Content (ATT&CK-mapping quality demos, honest
eval scorecards) as the trust builder. **Unvalidated.**

## Retention hypothesis

Retention comes from (a) privacy lock-in (already in their workflow/self-host),
(b) steadily improving model quality via the experiment loop, (c) usefulness in
daily triage. **Unvalidated.**

## Security / privacy expectations (a feature, not overhead)

Self-host + no content logging + safe-by-default is the core selling point for the
target customer; see `docs/security.md`. This is where a small private product can
credibly beat a big public one.

## Legal / licensing

- Code: Apache-2.0. Base model: **Gemma terms apply** — any redistribution or
  hosted offering of a Gemma-derived model must comply with Google's Gemma license
  (review before any paid distribution).
- Training data policy (`DATA_LICENSES.md`): CC-BY-authored content only; **no
  HTB/THM** or scraped proprietary content. Preserve this — it is what keeps a
  commercial offering clean.
- Output disclaimer: outputs may be wrong; not professional security advice.

## MVP monetization path (smallest honest step)

1. **Gate:** get one specialized model past its evaluation criteria (real quality).
2. Ship the free local tier; measure adoption + inference cost under real use.
3. Stand up a single hosted Pro instance behind auth + rate limiting (both already
   built); offer it to a handful of design-partner practitioners.
4. **Only then** price it, using measured infra cost and observed willingness to pay.

Do not build billing, teams, or SSO before step 3 produces paying interest.
