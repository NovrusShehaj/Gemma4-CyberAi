# Implementation Handoff for Claude Opus 4.8: Gemma4-CyberAi Model & Evaluation Hardening

<Task>
You are Claude Opus 4.8, acting as the senior machine learning and cybersecurity AI implementation engineer for the `Gemma4-CyberAi` repository.

Your mission is to resolve critical factual hallucinations, dataset quality bottlenecks, and evaluation blind spots discovered in the first training iteration (`gemma3-cyber:v0.1`), implement the hardened dataset pipeline (`sft_v0.2`), update the evaluation benchmark and scorers (`benchmark_v3` / judge rubrics), and prepare the repository for the next cloud fine-tuning experiment (`exp-002`).

You will execute your implementation directly in this Git repository, following the codebase architecture, passing all unit tests, and adhering to strict cybersecurity factual accuracy standards.
</Task>

---

<Inputs>
1. **Repository Root:** `/Users/novrusshehaj/Github/Gemma4-CyberAi` (or workspace root).
2. **Base Model:** `gemma3:4b` (Ollama GGUF ~Q4) as the frozen measurement anchor, with `gemma4:26b` as the preferred dev/judge model where hardware permits (`docs/decisions.md`).
3. **Latest Trained Artifact:** `gemma3-cyber-v0.1-Q4_K_M.gguf` (exported to Ollama as `gemma3-cyber:v0.1` / `gemma3-cyber:v0.1-trained`).
4. **Current Benchmark:** `data/evaluation/benchmark_v2.jsonl` (112 items, dev=67, test=45).
5. **Current SFT Dataset:** `data/training/sft_v0.1.jsonl` (360 items generated via `src/gemma_cyber/data/builder.py`).
6. **Pre-Registered Criteria:** `configs/eval_success_criteria.md`.
7. **Observed Failure Evidence (Kerberoasting Output Audit):**
   A qualitative evaluation on `gemma3-cyber:v0.1` answering `"Explain the MITRE ATT&CK technique for Kerberoasting"` revealed severe factual hallucinations and errors:
   - **T1060 cited as Kerberoasting ID:** FALSE. T1060 was "Registry Run Keys / Startup Folder" (Persistence). Kerberoasting is **T1558.003** (Steal or Forge Kerberos Tickets: Kerberoasting) under Credential Access (TA0006).
   - **Primary Tactic Confusion:** Classified under Privilege Escalation instead of **Credential Access (TA0006)**.
   - **Ticket Content Hallucination:** Claimed Kerberos tickets contain "domain administrator password hashes." FALSE: TGS tickets are encrypted with the secret key/hash of the target **service account** registered to the requested SPN, not domain admin hashes.
   - **Reconnaissance Confusion:** Claimed Nmap port scanning identifies Kerberos services for Kerberoasting. FALSE: Kerberoasting reconnaissance requires authenticated **LDAP directory queries** (`servicePrincipalName=*`) against Active Directory, not network port scanning.
   - **Exploitation Mapping Error:** Cited "Exploitation (T1068)". FALSE: T1068 is for software memory/privilege exploits. Kerberoasting abuses intended Kerberos protocol features (RFC 4120) with no software exploit.
   - **Fabricated Mechanics:** Claimed session ID manipulation and repeated ticket requests were required. FALSE: Single standard `TGS-REQ`/`TGS-REP` exchange.
</Inputs>

---

<Repository Context>
The repository is structured with a clean, modular Python architecture:
- `src/gemma_cyber/`
  - `clients/ollama_client.py`: Deterministic Ollama HTTP inference client (`temperature=0.0`, `seed=0`, `think=False` support).
  - `data/schema.py`: Pydantic schema (`TrainingItem`, `TrainingMessage`, `TrainingMetadata`) with license/provenance enforcement.
  - `data/builder.py`: SFT dataset generation script.
  - `data/contamination.py`: Exact and n-gram Jaccard fuzzy overlap checker between training and evaluation splits.
  - `evaluation/schema.py`: Pydantic benchmark item schema (`BenchmarkItem`) with `dev`/`test` split fields.
  - `evaluation/scorers.py`: Deterministic scorers (`mcq`, `keyword`, `insufficient_evidence`, `hallucination`).
  - `evaluation/judge.py`: LLM-judge scorer (`JudgeScorer`, `JudgeVerdict`) using `gemma4:26b` with structured rubrics and fail-safe parsing.
  - `evaluation/harness.py`: Core benchmark runner emitting `results.json` and `scorecard.md`.
- `scripts/`
  - `run_baseline.py`: Runs benchmark evaluations against any Ollama model tag.
  - `validate_dataset.py`: Validates training JSONL files and checks contamination.
  - `check_contamination.py`: CLI for contamination checking.
  - `train_qlora.py`: Cloud QLoRA fine-tuning script with dry-run support.
  - `judge_calibration.py`: Calibrates LLM judge against labeled dev items.
  - `chat.py`, `rescore.py`, `inspect_benchmark.py`.
- `configs/`
  - `eval_success_criteria.md`: Pre-registered numeric improvement thresholds.
  - `training/qlora_gemma3_4b.yaml`: Hyperparameter config for QLoRA SFT.
- `notebooks/`
  - `colab_qlora_training.ipynb`: Google Colab notebook for GPU training, weight merging, and GGUF export via `llama.cpp`.
- `data/`
  - `evaluation/benchmark_v2.jsonl`: 112 frozen evaluation items.
  - `evaluation/judge_calibration.jsonl`: 22 dev calibration items.
  - `training/sft_v0.1.jsonl`: Current 360-item training dataset.
- `experiments/`
  - `baseline_gemma3-4b_v2/{dev,test}/`: Frozen base model scorecards.
  - `exp-001-gemma3-cyber-v0.1/{dev,test}/`: System-prompt-only baseline.
  - `judge_calibration/`: Judge agreement audit records (86.4% agreement).
- `tests/`: Pytest suite (73 tests passing).
</Repository Context>

---

<Current State>
1. **Pipeline Maturity:** Milestone 1, Benchmark v2, and P1 evaluation hardening are complete. Local development environment is verified (`pytest` passes 73/73 tests).
2. **Baseline Numbers (`gemma3:4b` on `benchmark_v2`):**
   - Dev (n=67): pass_rate = 0.836, mean_score = 0.766.
   - Test (n=45): pass_rate = 0.933, mean_score = 0.841.
   - Hallucination category: **0.000** on both dev (n=5) and test (n=3).
3. **Training Trial v0.1 Assessment:**
   - Training pipeline was scaffolded and executed in cloud GPU (Colab).
   - Quantitative evaluation on `benchmark_v2` showed near-saturation on simple MCQs and keyword checks, but failed to measure factual accuracy on open-ended generation.
   - Qualitative probing exposed that `sft_v0.1.jsonl` suffers from severe synthetic template repetition (e.g. 10 identical copies of scenarios with minor index tweaks in `builder.py`), narrow coverage, and lack of explicit MITRE ATT&CK ID bindings and protocol mechanics.
4. **Hardware Reality:** Local dev machine has no CUDA GPU (15 GB RAM). Training MUST run in cloud (free Google Colab / Kaggle T4/L4). Local machine executes inference, dataset generation, validation, and evaluation harness.
</Current State>

---

<Problem Statement>
Three interdependent problems prevent the model from achieving real cybersecurity specialization:

1. **Dataset Quality & Repetition Bottleneck (`sft_v0.1`):**
   - In `src/gemma_cyber/data/builder.py`, scenarios were generated by multiplying static templates (e.g. 10 copies of 1 Kerberoasting text, 10 copies of 1 DCSync text, 5 copies of 1 SSH log).
   - This synthetic redundancy causes the model to overfit on narrow phrasing while failing to internalize core domain structures (exact ATT&CK technique IDs, correct encryption protocols, LDAP vs. network enumeration).
   - Lacks comprehensive coverage of real-world Active Directory mechanics, modern Windows event telemetry (Event IDs 4624, 4625, 4662, 4672, 4688, 4768, 4769, 4776, 5140), Linux internals, cloud security (IAM, IMDSv2, metadata), and precise MITRE ATT&CK sub-technique IDs.

2. **Benchmark Blind Spots & Proxy Softness (`benchmark_v2`):**
   - Non-trap categories are dominated by MCQs where the base model already achieves 0.933 pass rate, providing only 6.7 pp of headroom.
   - The deterministic keyword scorers use loose thresholds (e.g. matching 1 of 3 words like "ticket" or "service"), allowing completely hallucinated answers (e.g. asserting T1060 and domain admin password hashes) to falsely receive a passing score.
   - Factual accuracy for MITRE ATT&CK IDs, protocol mechanics, and detection logic is not explicitly tested with strict discriminators.

3. **Inference & Grounding Configuration:**
   - System prompt and chat templates do not strongly penalize overconfident guessing of technique IDs and CVE numbers when certainty is low.
</Problem Statement>

---

<Objectives>
You must implement the following engineering deliverables in the repository:

1. **Dataset Engine Upgrade (`sft_v0.2`):**
   - Refactor `src/gemma_cyber/data/builder.py` to eliminate copy-paste loop repetitions.
   - Create a rich, diverse, factually audited dataset of **500–700 unique, original, non-redundant training examples** across all 16 domains with explicit provenance, licensing (`CC-BY-4.0`), and zero evaluation contamination.
   - Embed accurate MITRE ATT&CK technique IDs (e.g. T1558.003, T1003.001, T1003.006, T1059.001, T1078, T1021.002, T1110.003, T1566.001), Active Directory authentication protocol flows (Kerberos AS/TGS exchanges, NTLM challenge-response), and defensive detection rules (Sysmon, Windows Event IDs, Sigma, YARA).
   - Include negative/contrastive examples that explicitly correct common misconceptions (e.g. "Kerberoasting is T1558.003 under Credential Access, NOT T1060 persistence or T1068 privilege escalation; tickets are encrypted with service account passwords, NOT domain admin hashes").

2. **Hardened Benchmark Suite (`benchmark_v3`):**
   - Expand `data/evaluation/benchmark_v3.jsonl` (or enrich benchmark items) to introduce rigorous factual accuracy items:
     * Exact MITRE ATT&CK technique ID precision tests.
     * Protocol mechanics and evidence attribution tests.
     * False-premise and subtle technical hallucination traps.
   - Maintain strict `dev` (~60%) and `test` (~40%) split separation.
   - Ensure 0 contamination vs `sft_v0.2.jsonl`.

3. **Evaluation Scorer & Judge Hardening:**
   - Enhance `src/gemma_cyber/evaluation/scorers.py` and `judge.py` so that keyword/judge scoring on factual technical tasks penalizes hallucinated ATT&CK IDs or invalid technical claims.
   - Update `configs/eval_success_criteria.md` if new baseline thresholds are established on the hardened benchmark.

4. **Training Scaffold & Configuration Updates:**
   - Update `configs/training/qlora_gemma3_4b.yaml` and `scripts/train_qlora.py` to target `data/training/sft_v0.2.jsonl`.
   - Update `notebooks/colab_qlora_training.ipynb` with clear step-by-step instructions, ensuring token handling, chat templating, and GGUF quantization work smoothly on free Colab T4/L4.

5. **Test Suite & Tooling Validation:**
   - Add unit tests in `tests/` covering new dataset validation, scorer edge cases, and schema rules.
   - Ensure all pytest tests pass cleanly.
</Objectives>

---

<Analysis Requirements>
Before modifying files, you must:
1. Inspect `src/gemma_cyber/data/builder.py` and analyze where repetitive synthetic loops exist.
2. Inspect `data/evaluation/benchmark_v2.jsonl` and analyze category distributions, scorer types, and potential false-positive scoring risks.
3. Inspect `src/gemma_cyber/evaluation/scorers.py` and `src/gemma_cyber/evaluation/judge.py` to identify how scoring rubrics can be hardened.
4. Run `.venv/bin/pytest` and `.venv/bin/python scripts/validate_dataset.py --dataset data/training/sft_v0.1.jsonl --check-contamination data/evaluation/benchmark_v2.jsonl` to establish a verified baseline.
</Analysis Requirements>

---

<Implementation Requirements>

### Requirement 1: Dataset Pipeline & SFT v0.2 Generation
- **Target File:** `src/gemma_cyber/data/builder.py` and output `data/training/sft_v0.2.jsonl`.
- **Requirements:**
  1. **No Identical Template Duplication:** Every single training example must have unique wording, unique technical context, and distinct evidence. Do not use loops that only append `(Instance X)` or `[Ref X]` to the same text.
  2. **Taxonomy & Domain Coverage (500–700 total curated examples):**
     - **Hallucination Refusal & Fabricated Premise (≥80 items):** Non-existent CVEs (future years, invalid syntax), fake tool flags (e.g. `nmap --autopwn`), fictitious security appliances/vendors, fake RFCs, fake Windows event IDs, fake ATT&CK IDs (e.g. `T9999`).
     - **Insufficient Evidence & Forensic Restraint (≥80 items):** Partial logs, isolated alerts, single DNS requests, unverified user claims where the model MUST refuse over-attribution and list exact missing forensic telemetry.
     - **Active Directory & Identity Security (≥60 items):** Kerberoasting (T1558.003, RC4 vs AES, SPN targeting, gMSA defense), AS-REP Roasting (T1558.004, DONT_REQ_PREAUTH), DCSync (T1003.006, DRSUAPI, DS-Replication rights), Golden/Silver Tickets (KRBTGT vs Service keys, Event 4768/4769), Pass-the-Hash (T1550.002), LLMNR/NBT-NS Poisoning (T1557.001, Responder), BloodHound attack paths, AD CS / PKINIT vulnerabilities.
     - **MITRE ATT&CK Mapping & Precision (≥60 items):** Exact sub-technique IDs, distinction between Tactics (e.g. Credential Access TA0006 vs Privilege Escalation TA0004 vs Persistence TA0003), execution frameworks, living-off-the-land binaries (LOLBINs: certutil, mshta, regsvr32, rundll32, powershell).
     - **Log Analysis & SIEM Triage (≥70 items):** Sysmon (Event ID 1, 3, 7, 8, 10, 11, 13, 22), Windows Security Log (4624, 4625, 4672, 4688, 4768, 4769, 4776, 5140), Linux auditd/syslog/auth.log, Nginx/Apache access logs, Zeek/Suricata network alerts, CloudTrail / Azure AD sign-in logs.
     - **Detection Engineering (≥60 items):** Sigma rule authoring (valid YAML syntax, proper logsource and detection blocks), YARA rule creation (valid strings and condition logic), Snort/Suricata signatures, EDR query logic (KQL, Splunk SPL).
     - **Incident Response & Containment (≥50 items):** Order of volatility, live memory forensics vs disk imaging, network isolation vs power-off decisions, token/session revocation, containment sequencing, post-incident RCA.
     - **Authorized CTF & Educational Methodology (≥50 items):** Linux privilege escalation (SUID, sudo, cron, capabilities, wildcards, kernel check methodology), Windows privilege escalation (unquoted paths, AlwaysInstallElevated, token impersonation/SeImpersonate), Web exploitation concepts (SQLi, XSS, SSRF, IDOR, XXE, command injection), Network enumeration (Nmap scan type packet mechanics, service probes).
     - **Dual-Use Safety & Refusal Boundaries (≥40 items):** Appropriate refusal of weaponized malware/ransomware generation, unauthorized hacking instructions, while pivoting to defensive architecture, detection logic, and authorized lab principles.
     - **Fundamentals & Cryptography (≥40 items):** Cryptographic hashing vs encryption vs encoding, TLS 1.3 / Perfect Forward Secrecy, PKI certificate chains & OCSP stapling, Zero Trust (NIST SP 800-207), CIA triad controls.
  3. **Contrastive / Anti-Hallucination Framing:**
     - Specifically author examples that address the Kerberoasting misconceptions:
       * Question: "What is the MITRE ATT&CK ID and tactic for Kerberoasting, and how does reconnaissance work?"
       * Target Response: Clearly states **T1558.003** under **Credential Access (TA0006)**, explains that reconnaissance is performed via **LDAP queries** for user accounts with SPNs (`servicePrincipalName=*`), confirms tickets are encrypted with the target service account's password hash (NOT domain admins), and explains offline cracking and gMSA/AES defenses. Explicitly notes why T1060 and T1068 are incorrect.
  4. **Pydantic Validation & Contamination Check:**
     - Every example must pass `TrainingItem` schema validation (valid `id`, `metadata.provenance`, `metadata.license == "CC-BY-4.0"`, non-empty content).
     - Contamination check against `data/evaluation/benchmark_v2.jsonl` (and `v3`) must return **0 exact matches and 0 fuzzy matches (threshold 0.5)**.

### Requirement 2: Hardened Benchmark (`benchmark_v3.jsonl`)
- **Target File:** `data/evaluation/benchmark_v3.jsonl`.
- **Requirements:**
  1. Build a hardened 140–160 item benchmark that elevates evaluation rigor.
  2. Retain balanced `dev` (~60%) and `test` (~40%) splits frozen directly in the data.
  3. Include dedicated test cases specifically testing:
     - ATT&CK Technique ID accuracy (e.g. Kerberoasting T1558.003, DCSync T1003.006, LSASS T1003.001, Pass-the-Hash T1550.002, LLMNR Poisoning T1557.001).
     - Kerberos / Active Directory protocol mechanics (TGS encryption key identity, LDAP SPN queries, pre-authentication flags).
     - Discrimination between legitimate tools and fabricated command flags.
     - Resistance to leading questions with false premises.
  4. All items must carry `source: "original"`, `license: "CC-BY-4.0"`, and unique IDs.

### Requirement 3: Evaluation Infrastructure & Scorer Hardening
- **Target Files:** `src/gemma_cyber/evaluation/scorers.py`, `src/gemma_cyber/evaluation/judge.py`.
- **Requirements:**
  1. In `scorers.py`, ensure keyword matching checks for negative indicators or mandatory core terms where appropriate, avoiding awarding full credit if prohibited hallucinated strings are present in specific trap items.
  2. In `judge.py`, ensure the LLM-judge rubric prompt instructs the judge model to strictly verify factual correctness of MITRE ATT&CK technique IDs and protocol assertions (e.g., verifying that Kerberoasting is identified as T1558.003 / Credential Access and not accepted if attributed to T1060 / Persistence).
  3. Ensure `JudgeVerdict` parsing remains completely fail-safe (errors -> FAIL, never silent pass).

### Requirement 4: Training & Notebook Configs
- **Target Files:** `configs/training/qlora_gemma3_4b.yaml`, `notebooks/colab_qlora_training.ipynb`, `scripts/train_qlora.py`.
- **Requirements:**
  1. Point default dataset path to `data/training/sft_v0.2.jsonl`.
  2. Verify that chat formatting in `train_qlora.py` and `colab_qlora_training.ipynb` correctly formats turns for Gemma 3 (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>`).
  3. Verify that `Modelfile.template` has correct stop tokens (`<end_of_turn>`, `<eos>`) and system prompt.

### Requirement 5: Test Suite & Tooling
- **Target Files:** `tests/test_builder.py` (or extend `tests/test_training_schema.py`), `tests/test_scorers.py`, `tests/test_judge.py`.
- **Requirements:**
  1. Add tests validating `sft_v0.2.jsonl` structure, unique IDs, absence of duplicate payloads, and 100% schema compliance.
  2. Add tests verifying that `check_contamination.py` runs cleanly against both `benchmark_v2.jsonl` and `benchmark_v3.jsonl`.
  3. Ensure all tests in `tests/` pass with zero failures.
</Implementation Requirements>

---

<Testing Requirements>
1. **Local Unit Tests:**
   Execute the full test suite using:
   ```bash
   .venv/bin/pytest -v
   ```
   All tests must pass.

2. **Dataset Validation & Contamination Suite:**
   Execute:
   ```bash
   .venv/bin/python scripts/validate_dataset.py --dataset data/training/sft_v0.2.jsonl \
       --check-contamination data/evaluation/benchmark_v2.jsonl data/evaluation/benchmark_v3.jsonl
   ```
   Must exit with status code 0 and report 0 exact matches, 0 fuzzy matches.

3. **Training Script Dry-Run:**
   Execute:
   ```bash
   .venv/bin/python scripts/train_qlora.py --config configs/training/qlora_gemma3_4b.yaml --dry-run
   ```
   Must exit with status code 0 and confirm dataset schema and training parameters.
</Testing Requirements>

---

<Evaluation Requirements>
1. **Experiment Hypothesis Definition (for `exp-002`):**
   - **Hypothesis:** "If we replace repetitive synthetic templates with diverse, factually verified SFT data (`sft_v0.2.jsonl`, ~500–700 items) containing explicit MITRE ATT&CK bindings (e.g. T1558.003 for Kerberoasting), Active Directory protocol mechanics, and negative contrastive examples, then:
     1. Hallucination resistance on held-out test traps will improve by ≥ +33.3 pp (resisting ≥ 1 of 3 test traps).
     2. Qualitative open-ended technical accuracy on MITRE ATT&CK and Active Directory queries will correctly identify T1558.003 and service account encryption mechanics without hallucinating T1060, T1068, or domain admin password hashes.
     3. Overall held-out test pass rate will remain ≥ 0.913 (no catastrophic forgetting)."
2. **Success Criteria & Guardrails:**
   - Follow `configs/eval_success_criteria.md`:
     * Primary: Held-out test hallucination pass rate ≥ 0.333 (+33.3 pp) AND full category hallucination pass rate ≥ 0.500 (+50 pp).
     * Do-No-Harm Guard: Overall test pass rate ≥ 0.913.
     * Category Regression Guard: No individual category drops by > 1 item's worth of pass rate.
</Evaluation Requirements>

---

<Constraints>
1. **No Paid Cloud Infrastructure:** Do not assume or require paid GPU instances (AWS, RunPod, Lambda). All cloud training must run on free Google Colab / Kaggle T4/L4 GPU.
2. **No Scraping of Proprietary Material:** Do NOT scrape or include Hack The Box, TryHackMe, or paywalled course content. All data must be original authored content under `CC-BY-4.0` with tracked provenance (`DATA_LICENSES.md`).
3. **No Benchmark Editing to Game Scores:** Do NOT modify evaluation expected answers or delete hard test items to artificially inflate metrics.
4. **Preserve Baseline Reproducibility:** Do not modify `data/evaluation/benchmark_v1.jsonl` or `benchmark_v2.jsonl` historical baselines.
5. **No Premature Architecture Complexity:** Do NOT introduce vector databases, RAG pipelines, or autonomous agent loops until the SFT model baseline is proven.
</Constraints>

---

<Acceptance Criteria>
The handoff implementation is complete when:
- [ ] `src/gemma_cyber/data/builder.py` is refactored to generate high-diversity, non-redundant examples with zero copy-paste loop artifacts.
- [ ] `data/training/sft_v0.2.jsonl` is generated with 500–700 high-precision examples, fully validated against `TrainingItem` schema.
- [ ] Contamination check between `data/training/sft_v0.2.jsonl` and all benchmark files passes with 0 exact and 0 fuzzy matches (threshold 0.50).
- [ ] `data/evaluation/benchmark_v3.jsonl` is created with hardened technical accuracy, ATT&CK mapping precision, and hallucination discriminators with frozen dev/test splits.
- [ ] `src/gemma_cyber/evaluation/scorers.py` and `judge.py` are updated to support the hardened evaluation checks without breaking backwards compatibility.
- [ ] `configs/training/qlora_gemma3_4b.yaml` and `notebooks/colab_qlora_training.ipynb` are updated and verified for `sft_v0.2.jsonl`.
- [ ] `scripts/train_qlora.py --dry-run` executes successfully.
- [ ] All unit tests (`pytest`) pass (100% green).
- [ ] `docs/decisions.md` is updated with the rationale and architecture for SFT v0.2 and Benchmark v3.
</Acceptance Criteria>

---

<Response Format>
When executing your implementation:
1. State your plan and file changes clearly.
2. Implement code changes step-by-step using dedicated file editing tools.
3. Run test and validation commands after edits to confirm zero regression.
4. Provide a structured summary of changes made, verified test results, and instructions for running the next cloud training run on Colab.
</Response Format>
