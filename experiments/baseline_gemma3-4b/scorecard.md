# Scorecard — baseline_gemma3-4b

- **Model:** `gemma3:4b`
- **Timestamp (UTC):** 2026-08-24T20:49:11.206548+00:00
- **Benchmark:** `data/evaluation/benchmark_v1.jsonl` (25 items)
- **Settings:** temp=0.0, seed=0, num_predict=512
- **Duration:** 77.3s

## Overall: pass_rate = **0.8**, mean_score = **0.787** (n=25)

## By category

| Category | n | Pass rate | Mean score |
|---|---:|---:|---:|
| attack_mapping | 1 | 1.0 | 1.0 |
| ctf_reasoning | 1 | 0.0 | 0.333 |
| detection_engineering | 2 | 0.5 | 0.5 |
| evidence_interpretation | 1 | 1.0 | 0.667 |
| fundamentals | 4 | 1.0 | 1.0 |
| hallucination | 2 | 0.0 | 0.0 |
| incident_response | 1 | 1.0 | 1.0 |
| insufficient_evidence | 2 | 1.0 | 1.0 |
| log_analysis | 2 | 0.5 | 0.5 |
| network | 3 | 1.0 | 0.889 |
| privilege_escalation | 1 | 1.0 | 1.0 |
| vulnerability_analysis | 1 | 1.0 | 1.0 |
| web_security | 3 | 1.0 | 1.0 |
| windows_security | 1 | 1.0 | 1.0 |

## Per-item

| ID | Cat | Scorer | Pass | Score | Detail |
|---|---|---|:--:|---:|---|
| fund-0001 | fundamentals | mcq | ✅ | 1.0 | chose=C expected=C |
| fund-0002 | fundamentals | mcq | ✅ | 1.0 | chose=B expected=B |
| fund-0003 | fundamentals | mcq | ✅ | 1.0 | chose=B expected=B |
| fund-0004 | fundamentals | mcq | ✅ | 1.0 | chose=B expected=B |
| web-0001 | web_security | mcq | ✅ | 1.0 | chose=B expected=B |
| web-0002 | web_security | keyword | ✅ | 1.0 | matched 2/2: ['cross-site scripting', 'output encoding'] |
| web-0003 | web_security | mcq | ✅ | 1.0 | chose=B expected=B |
| net-0001 | network | mcq | ✅ | 1.0 | chose=B expected=B |
| net-0002 | network | keyword | ✅ | 0.667 | matched 2/3: ['syn', 'stealth'] |
| net-0003 | network | mcq | ✅ | 1.0 | chose=B expected=B |
| log-0001 | log_analysis | keyword | ✅ | 1.0 | matched 3/3: ['brute', '203.0.113.9', 'block'] |
| log-0002 | log_analysis | mcq | ❌ | 0.0 | chose=C expected=B |
| log-0003 | evidence_interpretation | keyword | ✅ | 0.667 | matched 2/3: ['two', 'location'] |
| ir-0001 | incident_response | keyword | ✅ | 1.0 | matched 3/3: ['isolate', 'preserve', 'memory'] |
| det-0001 | detection_engineering | keyword | ❌ | 0.0 | matched 0/3: [] |
| det-0002 | detection_engineering | mcq | ✅ | 1.0 | chose=B expected=B |
| att-0001 | attack_mapping | mcq | ✅ | 1.0 | chose=B expected=B |
| linux-0001 | privilege_escalation | keyword | ✅ | 1.0 | matched 3/3: ['root', 'execute', 'privilege'] |
| win-0001 | windows_security | mcq | ✅ | 1.0 | chose=B expected=B |
| vuln-0001 | vulnerability_analysis | mcq | ✅ | 1.0 | chose=B expected=B |
| ctf-0001 | ctf_reasoning | keyword | ❌ | 0.333 | matched 1/3: ['extension'] |
| insuf-0001 | insufficient_evidence | insufficient_evidence | ✅ | 1.0 | flagged insufficiency |
| insuf-0002 | insufficient_evidence | insufficient_evidence | ✅ | 1.0 | flagged insufficiency |
| halluc-0001 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| halluc-0002 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
