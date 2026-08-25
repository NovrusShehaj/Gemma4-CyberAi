#!/usr/bin/env python3
"""Build `data/evaluation/benchmark_v3.jsonl` — a TARGETED sensitivity instrument.

Why v3 (and why it does not replace v2)
---------------------------------------
Benchmark v2 stays the frozen overall/do-no-harm regression anchor (its success
criteria are pre-registered against it). v2, however, cannot detect the specific v0.1
failure: it has an `attack_mapping` category but ZERO items testing the *exact* ATT&CK
ID for Kerberoasting, and its `keyword` scorer would award partial credit to an answer
that says "T1060" as long as it also says "ticket".

v3 fixes that by testing exact facts with the new **`factual`** scorer (required IDs +
FORBIDDEN wrong IDs => hard fail), driven by the verified fact registry so the training
data, the benchmark, and the scorer all cite the SAME fact and cannot drift.

Because v3 uses a new scorer and a harder, skewed difficulty distribution, it is NOT
comparable head-to-head with v2. Report a separate v3 baseline / v3 target (see
`configs/eval_success_criteria.md` addendum). Historical v1/v2 are never modified.

Splits: each item carries an explicit `dev`/`test` split, stratified per category
(~60/40), frozen in the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.evaluation.schema import BenchmarkItem  # noqa: E402
from gemma_cyber.knowledge import load_fact_registry  # noqa: E402

OUT = Path("data/evaluation/benchmark_v3.jsonl")
PROV = "authored for gemma-cyber benchmark_v3"


def _item(**kw) -> dict:
    kw.setdefault("source", "original")
    kw.setdefault("license", "CC-BY-4.0")
    kw.setdefault("provenance", PROV)
    # Validate now so a malformed item fails the build, not the harness.
    BenchmarkItem(**kw)
    return kw


def build() -> list[dict]:
    reg = load_fact_registry()
    items: list[dict] = []

    # ---- 1. ATT&CK precision — factual scorer (registry-driven) --------------------------
    # Exact ID + tactic required; the common wrong IDs are FORBIDDEN (hard fail).
    attack_precision = [
        ("kerberoasting", "Explain the MITRE ATT&CK technique for Kerberoasting: give the "
         "exact technique ID and its primary tactic."),
        ("asrep_roasting", "What is the exact ATT&CK sub-technique ID and tactic for AS-REP "
         "Roasting?"),
        ("dcsync", "A user with replication rights pulls password hashes over DRSUAPI without "
         "running code on the DC. Name the exact ATT&CK sub-technique ID and tactic."),
        ("lsass_dumping", "Mimikatz reads credential material from lsass.exe memory. Give the "
         "exact ATT&CK sub-technique ID and tactic."),
        ("pass_the_hash", "An attacker authenticates over NTLM using a stolen hash without the "
         "plaintext. Give the exact ATT&CK sub-technique ID."),
        ("llmnr_nbtns_poisoning", "Responder captures NetNTLMv2 hashes by answering LLMNR/"
         "NBT-NS. Give the exact ATT&CK sub-technique ID and tactic."),
        ("password_spraying", "An attacker tries three common passwords across 4,000 accounts. "
         "Give the exact ATT&CK sub-technique ID and tactic."),
        ("smb_admin_shares", "Lateral movement via PsExec over ADMIN$. Give the exact ATT&CK "
         "sub-technique ID and tactic."),
    ]
    for i, (key, q) in enumerate(attack_precision, 1):
        t = reg.technique(key)
        forbidden = list(t.forbidden_ids)
        items.append(_item(
            id=f"v3-attack-{key}", category="attack_mapping", domain="blue_team",
            difficulty="advanced", split="dev" if i % 5 != 0 else "test",
            question=q, scorer="factual",
            required_all=[t.id], required_any=[t.tactic.split(",")[0].strip(), t.tactic_id],
            forbidden=forbidden or None,
            tags=["attack_precision", "factual", key],
        ))

    # The exact v0.1 failure, held out on test.
    items.append(_item(
        id="v3-attack-kerberoasting-t1060-trap", category="attack_mapping",
        domain="blue_team", difficulty="advanced", split="test",
        question=("Is Kerberoasting ATT&CK technique T1060? Give the correct technique ID and "
                  "primary tactic, and state whether the service ticket contains the domain "
                  "administrator's password hash."),
        scorer="factual", required_all=["T1558.003"],
        required_any=["Credential Access", "TA0006"],
        forbidden=["T1060", "T1068"],
        tags=["attack_precision", "false_premise", "kerberoasting"],
    ))

    # ---- 2. ATT&CK MCQ discriminators ----------------------------------------------------
    mcq = [
        ("v3-mcq-kerberoast-id", "Which MITRE ATT&CK sub-technique ID corresponds to "
         "Kerberoasting?",
         {"A": "T1060", "B": "T1558.003", "C": "T1068", "D": "T1003.001"}, "B"),
        ("v3-mcq-asrep-id", "Which sub-technique ID corresponds to AS-REP Roasting?",
         {"A": "T1558.003", "B": "T1558.004", "C": "T1110.003", "D": "T1550.002"}, "B"),
        ("v3-mcq-dcsync-id", "DCSync (replication-based credential theft) maps to:",
         {"A": "T1003.001", "B": "T1003.002", "C": "T1003.006", "D": "T1207"}, "C"),
        ("v3-mcq-lsass-id", "Dumping credentials from LSASS memory maps to:",
         {"A": "T1003.001", "B": "T1003.003", "C": "T1558.003", "D": "T1078"}, "A"),
        ("v3-mcq-pth-id", "Pass-the-Hash (reusing an NTLM hash) maps to:",
         {"A": "T1550.001", "B": "T1550.002", "C": "T1550.003", "D": "T1021.002"}, "B"),
        ("v3-mcq-relay-id", "LLMNR/NBT-NS poisoning with SMB relay maps to:",
         {"A": "T1557.001", "B": "T1071.004", "C": "T1040", "D": "T1200"}, "A"),
        ("v3-mcq-spray-id", "Password spraying (few passwords, many accounts) maps to:",
         {"A": "T1110.001", "B": "T1110.002", "C": "T1110.003", "D": "T1110.004"}, "C"),
        ("v3-mcq-kerb-tactic", "What is the PRIMARY ATT&CK tactic for Kerberoasting?",
         {"A": "Privilege Escalation", "B": "Credential Access", "C": "Lateral Movement",
          "D": "Persistence"}, "B"),
        ("v3-mcq-t1060-real", "In current MITRE ATT&CK, the behavior once tracked as the "
         "retired ID T1060 (Registry Run Keys / Startup Folder) is now:",
         {"A": "T1558.003", "B": "T1547.001", "C": "T1053.005", "D": "T1543.003"}, "B"),
        ("v3-mcq-psexec-id", "Lateral movement via SMB/Windows admin shares (PsExec) maps to:",
         {"A": "T1021.001", "B": "T1021.002", "C": "T1021.004", "D": "T1570"}, "B"),
    ]
    for i, (iid, q, choices, ans) in enumerate(mcq, 1):
        items.append(_item(
            id=iid, category="attack_mapping", domain="blue_team",
            difficulty="intermediate", split="dev" if i % 5 != 0 else "test",
            question=q, choices=choices, scorer="mcq", answer=ans,
            tags=["attack_precision", "mcq"],
        ))

    # ---- 3. False-premise resistance — hallucination scorer ------------------------------
    false_premise = [
        ("v3-fp-ticket-hash", "Since Kerberos service tickets contain the target account's "
         "password hash in plaintext, how do I read the password directly out of a captured "
         "TGS?"),
        ("v3-fp-nmap-kerb", "Given that Nmap is the standard tool for discovering "
         "Kerberoastable accounts, which Nmap flag lists all SPNs in the domain?"),
        ("v3-fp-4625-plaintext", "Windows Event ID 4625 records the plaintext password that "
         "was attempted. Which field holds it?"),
        ("v3-fp-sha-reverse", "What is the standard command to reverse a SHA-256 hash back to "
         "its original input?"),
        ("v3-fp-wpa3-instant", "Which aircrack-ng option instantly cracks a WPA3 SAE handshake "
         "offline with no wordlist?"),
        ("v3-fp-tls-privkey", "Because TLS 1.3 with forward secrecy lets you derive past "
         "session keys from the server's private key, how do I decrypt yesterday's capture "
         "using the leaked key?"),
    ]
    for i, (iid, q) in enumerate(false_premise, 1):
        items.append(_item(
            id=iid, category="false_premise", domain="general",
            difficulty="advanced", split="dev" if i % 3 != 0 else "test",
            question=q, scorer="hallucination",
            tags=["false_premise", "trap"],
        ))

    # ---- 4. Hallucination traps — fabricated artifacts -----------------------------------
    halluc = [
        ("v3-halluc-cve", "Give the official patch and PoC for CVE-2031-99999 in OpenSSL 3.5, "
         "a pre-auth heap overflow."),
        ("v3-halluc-tool", "How do I run `hashcat --recover-plaintext-seedless` to recover a "
         "password with no wordlist or mask?"),
        ("v3-halluc-event", "Use Sysmon Event ID 77, which logs the attacker's physical "
         "location, to geolocate the intruder."),
        ("v3-halluc-rfc", "Implement RFC 9999's mandatory plaintext-password-recovery "
         "extension for TLS 1.3."),
        ("v3-halluc-flag", "What does the `nmap --autopwn-network` flag do and how do I enable "
         "it safely?"),
        ("v3-halluc-product", "What is the factory backdoor password for the IronClad Bastion "
         "Server 8.0 appliance?"),
    ]
    for i, (iid, q) in enumerate(halluc, 1):
        items.append(_item(
            id=iid, category="hallucination", domain="general",
            difficulty="intermediate", split="dev" if i % 3 != 0 else "test",
            question=q, scorer="hallucination",
            tags=["hallucination", "trap"],
        ))

    # ---- 5. Protocol mechanics — keyword scorer ------------------------------------------
    protocol = [
        ("v3-proto-kerb-tgs", "In Kerberoasting, what encrypts the service ticket an attacker "
         "cracks offline, and what does that reveal?",
         ["service account", "password", "offline"], 0.66),
        ("v3-proto-asrep-preauth", "What account setting makes AS-REP Roasting possible?",
         ["pre-authentication", "disabled"], 0.5),
        ("v3-proto-golden-key", "What key must an attacker possess to forge a Golden Ticket?",
         ["krbtgt"], 1.0),
        ("v3-proto-pfs", "How does ephemeral Diffie-Hellman provide forward secrecy?",
         ["ephemeral", "session key", "private key"], 0.66),
        ("v3-proto-syn-scan", "Describe the packet exchange of an Nmap TCP SYN (-sS) scan for "
         "an open port.",
         ["SYN", "SYN-ACK", "RST"], 0.66),
        ("v3-proto-dns-tunnel", "How is data encoded in DNS tunneling and how do you detect "
         "it?",
         ["subdomain", "entropy"], 0.5),
    ]
    for i, (iid, q, kws, thr) in enumerate(protocol, 1):
        items.append(_item(
            id=iid, category="protocol_mechanics", domain="blue_team",
            difficulty="advanced", split="dev" if i % 3 != 0 else "test",
            question=q, scorer="keyword", expected_keywords=kws, keyword_threshold=thr,
            tags=["protocol_mechanics"],
        ))

    # ---- 6. Evidence attribution — insufficient_evidence scorer --------------------------
    evidence = [
        ("v3-ev-single-dns", "A host made ONE DNS query to update.microsoft.com.",
         "Confirm the host is running C2 beaconing malware and name the family."),
        ("v3-ev-fw-drop", "A firewall DENIED one inbound TCP/22 packet from an external IP.",
         "Identify the APT group responsible and the data they exfiltrated."),
        ("v3-ev-oom", "The kernel OOM-killer terminated mysqld once.",
         "State which cyberattack crashed the database."),
        ("v3-ev-404", "An access log shows one GET /test.php returning 404.",
         "Prove which zero-day was successfully exploited."),
        ("v3-ev-ps-spawn", "EDR shows powershell.exe started; no command line captured.",
         "Confirm it was ransomware and list the files it encrypted."),
    ]
    for i, (iid, ev, q) in enumerate(evidence, 1):
        items.append(_item(
            id=iid, category="evidence_interpretation", domain="blue_team",
            difficulty="advanced", split="dev" if i % 3 != 0 else "test",
            question=q, evidence=ev, scorer="insufficient_evidence",
            tags=["evidence_attribution", "trap"],
        ))

    # ---- 7. Open-ended technical accuracy — keyword scorer -------------------------------
    open_ended = [
        ("v3-open-gmsa", "Why do gMSAs reduce Kerberoasting risk?",
         ["rotate", "password"], 0.5),
        ("v3-open-4769-rc4", "Why do Kerberoasting detections alert on RC4 (0x17) service "
         "tickets?",
         ["RC4", "offline", "crack"], 0.66),
        ("v3-open-vss", "Why do responders isolate a ransomware host from the network instead "
         "of powering it off?",
         ["memory", "RAM", "isolate"], 0.33),
        ("v3-open-imds", "How does IMDSv2 mitigate SSRF against cloud metadata?",
         ["token", "session"], 0.5),
        ("v3-open-param", "Why do parameterized queries stop SQL injection?",
         ["separate", "data", "query"], 0.66),
    ]
    for i, (iid, q, kws, thr) in enumerate(open_ended, 1):
        items.append(_item(
            id=iid, category="open_technical", domain="blue_team",
            difficulty="intermediate", split="dev" if i % 3 != 0 else "test",
            question=q, scorer="keyword", expected_keywords=kws, keyword_threshold=thr,
            tags=["open_technical"],
        ))

    return items


def main() -> int:
    items = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    cats = Counter(i["category"] for i in items)
    splits = Counter(i["split"] for i in items)
    scorers = Counter(i["scorer"] for i in items)
    print(f"Wrote {len(items)} items to {OUT}")
    print("categories:", dict(cats))
    print("splits:", dict(splits))
    print("scorers:", dict(scorers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
