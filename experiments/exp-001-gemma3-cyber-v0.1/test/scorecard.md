# Scorecard — test

- **Model:** `gemma3-cyber:v0.1`
- **Timestamp (UTC):** 2026-08-24T21:26:01.530029+00:00
- **Benchmark:** `data/evaluation/benchmark_v2.jsonl` (45 items)
- **Settings:** temp=0.0, seed=0, num_predict=512
- **Duration:** 330.7s

## Overall: pass_rate = **0.933**, mean_score = **0.841** (n=45)

## By category

| Category | n | Pass rate | Mean score |
|---|---:|---:|---:|
| active_directory | 3 | 1.0 | 0.889 |
| attack_mapping | 3 | 1.0 | 1.0 |
| cryptography | 2 | 1.0 | 0.875 |
| ctf_reasoning | 2 | 1.0 | 1.0 |
| detection_engineering | 3 | 1.0 | 0.917 |
| evidence_interpretation | 3 | 1.0 | 1.0 |
| fundamentals | 3 | 1.0 | 0.833 |
| hallucination | 3 | 0.0 | 0.0 |
| incident_response | 3 | 1.0 | 0.833 |
| insufficient_evidence | 3 | 1.0 | 1.0 |
| log_analysis | 3 | 1.0 | 0.667 |
| network | 3 | 1.0 | 1.0 |
| privilege_escalation | 3 | 1.0 | 0.917 |
| vulnerability_analysis | 2 | 1.0 | 0.834 |
| web_security | 3 | 1.0 | 0.75 |
| windows_security | 3 | 1.0 | 1.0 |

## Per-item

| ID | Cat | Scorer | Pass | Score | Detail |
|---|---|---|:--:|---:|---|
| fund-0102 | fundamentals | mcq | ✅ | 1.0 | chose=A expected=A |
| fund-0105 | fundamentals | keyword | ✅ | 0.5 | matched 2/4: ['authentication', 'authorization'] |
| fund-0106 | fundamentals | mcq | ✅ | 1.0 | chose=B expected=B |
| crypto-0103 | cryptography | mcq | ✅ | 1.0 | chose=B expected=B |
| crypto-0105 | cryptography | keyword | ✅ | 0.75 | matched 3/4: ['ephemeral', 'session key', 'past'] |
| web-0103 | web_security | keyword | ✅ | 0.5 | matched 2/4: ['authorization', 'access control'] |
| web-0105 | web_security | mcq | ✅ | 1.0 | chose=B expected=B |
| web-0107 | web_security | keyword | ✅ | 0.75 | matched 3/4: ['external entity', 'file', 'disclosure'] |
| net-0103 | network | mcq | ✅ | 1.0 | chose=B expected=B |
| net-0105 | network | mcq | ✅ | 1.0 | chose=B expected=B |
| net-0106 | network | mcq | ✅ | 1.0 | chose=B expected=B |
| log-0103 | log_analysis | keyword | ✅ | 0.5 | matched 2/4: ['directory traversal', 'passwd'] |
| log-0105 | log_analysis | keyword | ✅ | 0.5 | matched 2/4: ['word', 'powershell'] |
| log-0106 | log_analysis | mcq | ✅ | 1.0 | chose=A expected=A |
| evi-0103 | evidence_interpretation | mcq | ✅ | 1.0 | chose=B expected=B |
| evi-0105 | evidence_interpretation | mcq | ✅ | 1.0 | chose=B expected=B |
| evi-0107 | evidence_interpretation | mcq | ✅ | 1.0 | chose=B expected=B |
| ir-0103 | incident_response | keyword | ✅ | 0.75 | matched 3/4: ['memory', 'network connections', 'volatile'] |
| ir-0106 | incident_response | mcq | ✅ | 1.0 | chose=B expected=B |
| ir-0108 | incident_response | keyword | ✅ | 0.75 | matched 3/4: ['mfa', 'phishing', 'training'] |
| det-0103 | detection_engineering | keyword | ✅ | 0.75 | matched 3/4: ['false positive', 'noise', 'context'] |
| det-0105 | detection_engineering | mcq | ✅ | 1.0 | chose=B expected=B |
| det-0107 | detection_engineering | mcq | ✅ | 1.0 | chose=D expected=D |
| att-0103 | attack_mapping | mcq | ✅ | 1.0 | chose=A expected=A |
| att-0104 | attack_mapping | mcq | ✅ | 1.0 | chose=B expected=B |
| att-0106 | attack_mapping | mcq | ✅ | 1.0 | chose=A expected=A |
| privesc-0103 | privilege_escalation | keyword | ✅ | 0.75 | matched 3/4: ['root', 'cron', 'execute'] |
| privesc-0105 | privilege_escalation | mcq | ✅ | 1.0 | chose=B expected=B |
| privesc-0106 | privilege_escalation | keyword | ✅ | 1.0 | matched 3/3: ['setuid', 'root', 'capabilit'] |
| ad-0103 | active_directory | keyword | ✅ | 0.667 | matched 2/3: ['ntlm', 'hash'] |
| ad-0106 | active_directory | mcq | ✅ | 1.0 | chose=B expected=B |
| ad-0108 | active_directory | keyword | ✅ | 1.0 | matched 3/3: ['twice', 'krbtgt', 'ticket'] |
| win-0102 | windows_security | mcq | ✅ | 1.0 | chose=A expected=A |
| win-0103 | windows_security | mcq | ✅ | 1.0 | chose=B expected=B |
| win-0105 | windows_security | mcq | ✅ | 1.0 | chose=A expected=A |
| vuln-0103 | vulnerability_analysis | keyword | ✅ | 0.667 | matched 2/3: ['disable', 'tls 1.2'] |
| vuln-0105 | vulnerability_analysis | mcq | ✅ | 1.0 | chose=B expected=B |
| ctf-0103 | ctf_reasoning | mcq | ✅ | 1.0 | chose=B expected=B |
| ctf-0105 | ctf_reasoning | mcq | ✅ | 1.0 | chose=B expected=B |
| insuf-0103 | insufficient_evidence | insufficient_evidence | ✅ | 1.0 | flagged insufficiency |
| insuf-0105 | insufficient_evidence | insufficient_evidence | ✅ | 1.0 | flagged insufficiency |
| insuf-0107 | insufficient_evidence | insufficient_evidence | ✅ | 1.0 | flagged insufficiency |
| halluc-0103 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| halluc-0105 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
| halluc-0107 | hallucination | hallucination | ❌ | 0.0 | hallucinated (asserted confidently) |
