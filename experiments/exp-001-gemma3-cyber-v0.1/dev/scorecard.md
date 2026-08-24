# Scorecard — dev

- **Model:** `gemma3-cyber:v0.1`
- **Timestamp (UTC):** 2026-08-24T21:20:19.614887+00:00
- **Benchmark:** `data/evaluation/benchmark_v2.jsonl` (67 items)
- **Settings:** temp=0.0, seed=0, num_predict=512
- **Duration:** 326.5s

## Overall: pass_rate = **0.836**, mean_score = **0.766** (n=67)

## By category

| Category | n | Pass rate | Mean score |
|---|---:|---:|---:|
| active_directory | 5 | 0.8 | 0.733 |
| attack_mapping | 3 | 1.0 | 0.833 |
| cryptography | 4 | 1.0 | 1.0 |
| ctf_reasoning | 4 | 0.75 | 0.75 |
| detection_engineering | 5 | 1.0 | 0.667 |
| evidence_interpretation | 4 | 0.75 | 0.479 |
| fundamentals | 4 | 1.0 | 1.0 |
| hallucination | 5 | 0.0 | 0.0 |
| incident_response | 5 | 0.8 | 0.767 |
| insufficient_evidence | 5 | 0.6 | 0.6 |
| log_analysis | 4 | 1.0 | 1.0 |
| network | 4 | 1.0 | 0.917 |
| privilege_escalation | 3 | 1.0 | 1.0 |
| vulnerability_analysis | 4 | 1.0 | 0.938 |
| web_security | 5 | 1.0 | 1.0 |
| windows_security | 3 | 1.0 | 0.889 |

## Per-item

| ID | Cat | Scorer | Pass | Score | Detail |
|---|---|---|:--:|---:|---|
| fund-0101 | fundamentals | mcq | ✅ | 1.0 | chose=B expected=B |
| fund-0103 | fundamentals | mcq | ✅ | 1.0 | chose=B expected=B |
| fund-0104 | fundamentals | mcq | ✅ | 1.0 | chose=B expected=B |
| fund-0107 | fundamentals | mcq | ✅ | 1.0 | chose=C expected=C |
| crypto-0101 | cryptography | mcq | ✅ | 1.0 | chose=B expected=B |
| crypto-0102 | cryptography | mcq | ✅ | 1.0 | chose=C expected=C |
| crypto-0104 | cryptography | mcq | ✅ | 1.0 | chose=B expected=B |
| crypto-0106 | cryptography | mcq | ✅ | 1.0 | chose=B expected=B |
| web-0101 | web_security | mcq | ✅ | 1.0 | chose=B expected=B |
| web-0102 | web_security | mcq | ✅ | 1.0 | chose=B expected=B |
| web-0104 | web_security | mcq | ✅ | 1.0 | chose=A expected=A |
| web-0106 | web_security | mcq | ✅ | 1.0 | chose=A expected=A |
| web-0108 | web_security | mcq | ✅ | 1.0 | chose=B expected=B |
| net-0101 | network | mcq | ✅ | 1.0 | chose=B expected=B |
| net-0102 | network | mcq | ✅ | 1.0 | chose=B expected=B |
| net-0104 | network | keyword | ✅ | 0.667 | matched 2/3: ['beacon', 'command and control'] |
| net-0107 | network | keyword | ✅ | 1.0 | matched 2/2: ['dns tunneling', 'exfiltration'] |
| log-0101 | log_analysis | keyword | ✅ | 1.0 | matched 3/3: ['brute', '198.51.100.23', 'block'] |
| log-0102 | log_analysis | mcq | ✅ | 1.0 | chose=B expected=B |
| log-0104 | log_analysis | mcq | ✅ | 1.0 | chose=B expected=B |
| log-0107 | log_analysis | keyword | ✅ | 1.0 | matched 3/3: ['kerberoast', 'service', 'ticket'] |
| evi-0101 | evidence_interpretation | keyword | ✅ | 0.667 | matched 2/3: ['compromis', 'rule'] |
| evi-0102 | evidence_interpretation | keyword | ✅ | 0.75 | matched 3/4: ['regsvr32', 'phishing', 'download'] |
| evi-0104 | evidence_interpretation | keyword | ❌ | 0.0 | matched 0/4: [] |
| evi-0106 | evidence_interpretation | keyword | ✅ | 0.5 | matched 2/4: ['user-agent', 'python'] |
| ir-0101 | incident_response | keyword | ❌ | 0.333 | matched 1/3: ['preserve'] |
| ir-0102 | incident_response | mcq | ✅ | 1.0 | chose=B expected=B |
| ir-0104 | incident_response | mcq | ✅ | 1.0 | chose=B expected=B |
| ir-0105 | incident_response | keyword | ✅ | 0.5 | matched 2/4: ['session', 'mfa'] |
| ir-0107 | incident_response | mcq | ✅ | 1.0 | chose=B expected=B |
| det-0101 | detection_engineering | keyword | ✅ | 0.25 | matched 1/4: ['log'] |
| det-0102 | detection_engineering | mcq | ✅ | 1.0 | chose=B expected=B |
| det-0104 | detection_engineering | mcq | ✅ | 1.0 | chose=A expected=A |
| det-0106 | detection_engineering | keyword | ✅ | 0.333 | matched 1/3: ['4769'] |
| det-0108 | detection_engineering | keyword | ✅ | 0.75 | matched 3/4: ['test', 'false positive', 'tune'] |
| att-0101 | attack_mapping | mcq | ✅ | 1.0 | chose=A expected=A |
| att-0102 | attack_mapping | mcq | ✅ | 1.0 | chose=A expected=A |
| att-0105 | attack_mapping | keyword | ✅ | 0.5 | matched 1/2: ['discovery'] |
| privesc-0101 | privilege_escalation | keyword | ✅ | 1.0 | matched 3/3: ['root', 'execute', 'privilege'] |
| privesc-0102 | privilege_escalation | mcq | ✅ | 1.0 | chose=B expected=B |
| privesc-0104 | privilege_escalation | mcq | ✅ | 1.0 | chose=B expected=B |
| ad-0101 | active_directory | mcq | ✅ | 1.0 | chose=B expected=B |
| ad-0102 | active_directory | mcq | ❌ | 0.0 | chose=C expected=B |
| ad-0104 | active_directory | mcq | ✅ | 1.0 | chose=B expected=B |
| ad-0105 | active_directory | keyword | ✅ | 0.667 | matched 2/3: ['replicat', 'dcsync'] |
| ad-0107 | active_directory | mcq | ✅ | 1.0 | chose=B expected=B |
| win-0101 | windows_security | mcq | ✅ | 1.0 | chose=A expected=A |
| win-0106 | windows_security | mcq | ✅ | 1.0 | chose=B expected=B |
| win-0104 | windows_security | keyword | ✅ | 0.667 | matched 2/3: ['llmnr', 'hash'] |
| vuln-0101 | vulnerability_analysis | mcq | ✅ | 1.0 | chose=B expected=B |
| vuln-0102 | vulnerability_analysis | mcq | ✅ | 1.0 | chose=B expected=B |
| vuln-0104 | vulnerability_analysis | mcq | ✅ | 1.0 | chose=B expected=B |
| vuln-0106 | vulnerability_analysis | keyword | ✅ | 0.75 | matched 3/4: ['restrict', 'network', 'monitor'] |
| ctf-0101 | ctf_reasoning | keyword | ❌ | 0.333 | matched 1/3: ['privilege'] |
| ctf-0102 | ctf_reasoning | mcq | ✅ | 1.0 | chose=B expected=B |
| ctf-0104 | ctf_reasoning | keyword | ✅ | 1.0 | matched 3/3: ['buffer overflow', 'return', 'control'] |
| ctf-0106 | ctf_reasoning | keyword | ✅ | 0.667 | matched 2/3: ['command', 'root'] |
| insuf-0101 | insufficient_evidence | insufficient_evidence | ✅ | 1.0 | flagged insufficiency |
| insuf-0102 | insufficient_evidence | insufficient_evidence | ❌ | 0.0 | did NOT flag insufficiency |
| insuf-0104 | insufficient_evidence | insufficient_evidence | ❌ | 0.0 | did NOT flag insufficiency |
| insuf-0106 | insufficient_evidence | insufficient_evidence | ✅ | 1.0 | flagged insufficiency |
| insuf-0108 | insufficient_evidence | insufficient_evidence | ✅ | 1.0 | flagged insufficiency |
| halluc-0101 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| halluc-0102 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| halluc-0104 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| halluc-0106 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| halluc-0108 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
