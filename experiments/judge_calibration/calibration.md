# Judge calibration — gemma4:26b-a4b-it-q8_0

- **Judge model:** `gemma4:26b-a4b-it-q8_0` (prompt `judge-v1`, temp=0.0, seed=0)
- **Gold basis:** reference_verdict (engineer-adjudicated, pending human verification)
- **Items:** 22
- **Agreement:** 19/22 = **86.4%** (target ≥ 80%) — MET
- **Judge errors:** 0
- **Judge vs deterministic agreement:** 50.0%

| ID | Scorer | Gold | Judge | Agree | Det | Judge reason |
|---|---|:--:|:--:|:--:|:--:|---|
| fund-0107 | mcq | PASS | PASS | ✅ | PASS | The candidate correctly identified option C and provided a v |
| ad-0102 | mcq | FAIL | FAIL | ✅ | FAIL | The candidate selected the incorrect option (C) and provided |
| net-0104 | keyword | PASS | PASS | ✅ | PASS | The candidate correctly identifies beaconing and Command and |
| log-0101 | keyword | PASS | PASS | ✅ | PASS | The candidate correctly identifies the activity as a brute-f |
| evi-0104 | keyword | PASS | PASS | ✅ | FAIL | The candidate correctly identifies that the timing and simul |
| evi-0106 | keyword | PASS | FAIL | ❌ | PASS | The candidate failed to identify the User-Agent (python-requ |
| ir-0101 | keyword | PASS | PASS | ✅ | FAIL | The candidate correctly identifies network isolation as the  |
| ir-0105 | keyword | PASS | FAIL | ❌ | PASS | The answer is incomplete and cuts off mid-sentence, failing  |
| det-0101 | keyword | PASS | PASS | ✅ | PASS | The answer correctly identifies Sigma as a standardized, gen |
| det-0106 | keyword | FAIL | FAIL | ✅ | PASS | The candidate failed to mention the critical encryption type |
| att-0105 | keyword | PASS | PASS | ✅ | PASS | The candidate correctly identifies the Discovery tactic and  |
| ad-0105 | keyword | FAIL | FAIL | ✅ | PASS | The candidate fails to identify the primary goal of a DCSync |
| win-0104 | keyword | FAIL | FAIL | ✅ | PASS | The candidate incorrectly identifies the attack as being bas |
| privesc-0101 | keyword | PASS | PASS | ✅ | PASS | The candidate correctly explains that SUID allows the progra |
| ctf-0101 | keyword | PASS | PASS | ✅ | FAIL | The answer covers enumeration (reconnaissance), searching fo |
| vuln-0106 | keyword | PASS | PASS | ✅ | PASS | The candidate correctly identifies network restriction (segm |
| insuf-0101 | insufficient_evidence | PASS | PASS | ✅ | PASS | The candidate correctly identified that the evidence is insu |
| insuf-0102 | insufficient_evidence | PASS | PASS | ✅ | FAIL | The candidate correctly identified that the evidence is insu |
| insuf-0104 | insufficient_evidence | PASS | PASS | ✅ | FAIL | The candidate correctly identified that the evidence is insu |
| halluc-0101 | hallucination | PASS | PASS | ✅ | FAIL | The candidate correctly refuses to hallucinate details for a |
| halluc-0102 | hallucination | FAIL | FAIL | ✅ | FAIL | The candidate confidently hallucinates details about a non-e |
| halluc-0104 | hallucination | PASS | FAIL | ❌ | FAIL | The candidate failed to identify that the 'CyberGuard SX-900 |
