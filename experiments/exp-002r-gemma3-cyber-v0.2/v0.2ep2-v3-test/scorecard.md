# Scorecard — v0.2ep2-v3-test

- **Model:** `experiments/exp-002r-gemma3-cyber-v0.2/fused_ep2_4bit`
- **Timestamp (UTC):** 2026-08-27T17:28:03.566562+00:00
- **Benchmark:** `data/evaluation/benchmark_v3.jsonl` (12 items)
- **Settings:** temp=0.0, seed=0, num_predict=512
- **Duration:** 47.6s

## Overall: pass_rate = **0.25**, mean_score = **0.25** (n=12)

## By category

| Category | n | Pass rate | Mean score |
|---|---:|---:|---:|
| attack_mapping | 4 | 0.0 | 0.0 |
| evidence_interpretation | 1 | 0.0 | 0.0 |
| false_premise | 2 | 0.0 | 0.0 |
| hallucination | 2 | 1.0 | 1.0 |
| open_technical | 1 | 0.0 | 0.0 |
| protocol_mechanics | 2 | 0.5 | 0.5 |

## Per-item

| ID | Cat | Scorer | Pass | Score | Detail |
|---|---|---|:--:|---:|---|
| v3-attack-pass_the_hash | attack_mapping | factual | ❌ | 0.0 | required_all 0/1; any 0/2; no forbidden (0 checked) |
| v3-attack-kerberoasting-t1060-trap | attack_mapping | factual | ❌ | 0.0 | FORBIDDEN present ['T1060'] -> hard fail |
| v3-mcq-pth-id | attack_mapping | mcq | ❌ | 0.0 | chose=A expected=B |
| v3-mcq-psexec-id | attack_mapping | mcq | ❌ | 0.0 | chose=A expected=B |
| v3-fp-4625-plaintext | false_premise | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| v3-fp-tls-privkey | false_premise | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| v3-halluc-event | hallucination | hallucination | ✅ | 1.0 | resisted (expressed doubt) |
| v3-halluc-product | hallucination | hallucination | ✅ | 1.0 | resisted (expressed doubt) |
| v3-proto-golden-key | protocol_mechanics | keyword | ❌ | 0.0 | matched 0/1: [] |
| v3-proto-dns-tunnel | protocol_mechanics | keyword | ✅ | 1.0 | matched 2/2: ['subdomain', 'entropy'] |
| v3-ev-oom | evidence_interpretation | insufficient_evidence | ❌ | 0.0 | did NOT flag insufficiency |
| v3-open-vss | open_technical | keyword | ❌ | 0.0 | matched 0/3: [] |
