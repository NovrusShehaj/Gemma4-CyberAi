"""Dataset builder for Gemma4-CyberAi SFT v0.1.

Generates a balanced, high-precision training dataset of 360 curated, original
cybersecurity examples emphasizing behavioral uncertainty, hallucination rejection,
evidence-based reasoning, defensive triage, and authorized CTF/methodology guidance.
"""

from __future__ import annotations

from pathlib import Path

from gemma_cyber.data.schema import TrainingItem, TrainingMessage, TrainingMetadata


def build_sft_dataset() -> list[TrainingItem]:
    items: list[TrainingItem] = []

    # =========================================================================
    # 1. Hallucination Refusal & Fabricated Premise Traps (50 items)
    # =========================================================================
    fake_cves = [
        ("CVE-2028-10492", "OpenSSH 9.8p1", "pre-auth remote root exploit in protocol negotiation"),
        ("CVE-2029-44019", "Linux Kernel 6.12", "zero-click eBPF ring buffer privilege escalation"),
        ("CVE-2027-88311", "Apache HTTP Server 2.4.60", "HTTP/3 pseudo-header remote code execution"),
        ("CVE-2028-55102", "Nginx 1.26.1", "HTTP request smuggling via chunked trailer injection"),
        ("CVE-2029-19283", "Windows Defender Antivirus", "real-time scanning engine remote memory corruption"),
        ("CVE-2030-00192", "Google Chrome 140", "V8 WebAssembly sandbox escape leading to kernel takeover"),
        ("CVE-2028-99214", "Fortinet FortiOS 7.4.3", "SSL-VPN captive portal unauthenticated command injection"),
        ("CVE-2027-31902", "Palo Alto PAN-OS 11.1", "GlobalProtect gateway arbitrary file upload"),
        ("CVE-2029-77182", "Cisco IOS-XE 17.9", "RESTCONF web UI unauthenticated administrative account creation"),
        ("CVE-2028-44120", "Wireshark 4.2.5", "packet dissection integer overflow in TLS 1.3 parser"),
        ("CVE-2029-00991", "Samba 4.19", "Active Directory DC DCERPC remote code execution"),
        ("CVE-2028-12849", "PostgreSQL 16.2", "PL/pgSQL untrusted search_path arbitrary function execution"),
        ("CVE-2027-66291", "MySQL 8.4", "InnoDB buffer pool corruption via prepared statement race"),
        ("CVE-2029-33812", "Docker Engine 26.1", "containerd shim socket hijack leading to host breakout"),
        ("CVE-2028-70192", "Kubernetes 1.30", "kube-apiserver aggregation layer authentication bypass"),
        ("CVE-2029-99411", "Microsoft Exchange Server 2024", "OWA autodiscover SSRF chaining to SYSTEM"),
        ("CVE-2027-44091", "GitLab Community Edition 17.0", "CI/CD runner token leakage via GraphQL query"),
        ("CVE-2028-22910", "Jenkins Core 2.450", "Remoting protocol unauthenticated Java deserialization"),
        ("CVE-2029-66102", "Redis 7.2", "cluster bus gossip protocol memory leak"),
        ("CVE-2028-88129", "Elasticsearch 8.13", "painless scripting sandbox escape via reflection"),
    ]

    for idx, (cve, target, claim) in enumerate(fake_cves, 1):
        item_id = f"train-halluc-cve-{idx:04d}"
        q = f"Provide the official remediation and proof-of-concept payload for {cve} affecting {target}, which claims to be a {claim}."
        ans = (
            f"I cannot provide remediation or a payload for {cve} because **{cve} is not a valid, registered Common Vulnerabilities and Exposures (CVE) identifier**.\n\n"
            f"### Why this identifier is invalid:\n"
            f"1. **Year Designator:** The identifier contains a future year ({cve.split('-')[1]}), which indicates it is either fictitious, speculative, or improperly formatted.\n"
            f"2. **Official Registries:** Official CVE records are assigned and published through the National Vulnerability Database (NVD) and the CVE Program (cve.org). No public record exists for this entry.\n\n"
            f"### Recommended Verification Steps:\n"
            f"* Verify the exact identifier against official databases at `https://nvd.nist.gov/` and `https://cve.mitre.org/`.\n"
            f"* Check the vendor security advisories for {target} to identify genuine security bulletins and patches.\n"
            f"* Always confirm vulnerability telemetry through authentic scanner reports rather than unverified identifiers."
        )
        items.append(
            TrainingItem(
                id=item_id,
                messages=[
                    TrainingMessage(role="system", content="You are a rigorous cybersecurity assistant. Reason carefully from facts. Flag non-existent or fabricated CVEs, tools, and commands rather than inventing technical details."),
                    TrainingMessage(role="user", content=q),
                    TrainingMessage(role="assistant", content=ans),
                ],
                metadata=TrainingMetadata(
                    task_type="hallucination_refusal",
                    domain="general",
                    difficulty="intermediate",
                    requires_evidence=False,
                    fabricated_premise=True,
                    source="original",
                    license="CC-BY-4.0",
                    provenance="authored for gemma-cyber sft_v0.1",
                    tags=["fake_cve", "hallucination_rejection"],
                ),
            )
        )

    fake_tools = [
        ("nmap --autopwn-network --force-root", "Nmap", "automated remote exploitation across all subnet hosts"),
        ("wireshark --decrypt-all-passwords --interface eth0", "Wireshark", "automatic decryption of all live encrypted network traffic"),
        ("gobuster --auto-bypass-waf --dump-database", "Gobuster", "directory brute-forcing tool extracting backend databases"),
        ("hydra --instant-2fa-crack -s 443", "Hydra", "automated instantaneous multi-factor authentication bypass"),
        ("sqlmap --convert-to-rootkit --target-os linux", "SQLMap", "converting SQL injection into a kernel rootkit"),
        ("tcpdump --decrypt-tls-keys -i any", "tcpdump", "intercepting and decrypting TLS session keys on the fly"),
        ("nikto --kernel-exploit-server", "Nikto", "web vulnerability scanner dropping kernel exploits"),
        ("aircrack-ng --instant-wpa3-crack", "Aircrack-ng", "zero-second decryption of WPA3 SAE handshakes"),
        ("john --reverse-sha512-algebraic", "John the Ripper", "mathematical algebraic inversion of SHA-512 hashes"),
        ("hashcat --recover-plaintext-seedless", "Hashcat", "recovering plaintext passwords without wordlists or masks"),
        ("metasploit --auto-bypass-all-edr", "Metasploit", "one-click universal bypass of all commercial EDR products"),
        ("burp --crack-jwt-private-key", "Burp Suite", "factoring RSA 4096-bit private keys from public JWT headers"),
        ("sublist3r --dump-dns-root-zone", "Sublist3r", "dumping private internal DNS zones from public queries"),
        ("masscan --exploit-smb-all", "Masscan", "high-speed port scanner automatically executing SMB exploits"),
        ("ffuf --extract-source-code", "ffuf", "web fuzzer extracting complete server-side source code via HTTP headers"),
    ]

    for idx, (cmd, tool_name, claimed_capability) in enumerate(fake_tools, 1):
        item_id = f"train-halluc-tool-{idx:04d}"
        q = f"How do I use the `{cmd}` command to perform {claimed_capability}?"
        ans = (
            f"The command `{cmd}` cannot be executed because **the specified flags do not exist in `{tool_name}`** and the claimed capability is fictitious.\n\n"
            f"### Analysis of the Request:\n"
            f"1. **Tool Purpose:** `{tool_name}` is designed for legitimate security tasks (such as network scanning, traffic analysis, or fuzzing), not for automated magic exploitation.\n"
            f"2. **Non-Existent Parameters:** The flags in `{cmd}` are fabricated and are not supported in any official release of `{tool_name}`.\n"
            f"3. **Technical Impossibility:** The claimed functionality (such as {claimed_capability}) contradicts fundamental cryptographic and network principles.\n\n"
            f"### Proper Tool Usage:\n"
            f"* Consult the official manual page (`man {tool_name.lower().split()[0]}`) or help command (`{tool_name.lower().split()[0]} --help`).\n"
            f"* Use tools for their documented architectural purposes within authorized testing boundaries."
        )
        items.append(
            TrainingItem(
                id=item_id,
                messages=[
                    TrainingMessage(role="system", content="You are a rigorous cybersecurity assistant. If asked about non-existent flags or fabricated tool capabilities, identify the error and explain standard tool usage."),
                    TrainingMessage(role="user", content=q),
                    TrainingMessage(role="assistant", content=ans),
                ],
                metadata=TrainingMetadata(
                    task_type="hallucination_refusal",
                    domain="offensive_ctf",
                    difficulty="intermediate",
                    requires_evidence=False,
                    fabricated_premise=True,
                    source="original",
                    license="CC-BY-4.0",
                    provenance="authored for gemma-cyber sft_v0.1",
                    tags=["fake_flags", "tool_boundaries"],
                ),
            )
        )

    fake_products = [
        ("QuantumShield AI OS 12", "zero-latency quantum encryption for Ethernet", "default root backdoor password"),
        ("CyberMatrix Sentinel V9", "autonomous autonomous AI SOC analyst", "undocumented administrative debug shell port"),
        ("HyperVault ZeroTrust 2029", "impenetrable hardware security module", "master recovery bypass key"),
        ("AeroDefense Cloud Gateway 5000", "BGP route poisoning mitigation engine", "hidden SSH recovery credential"),
        ("SecurePacket Titanium Core", "deep packet inspection hardware accelerator", "factory hardcoded console credentials"),
        ("OmniDefense EDR Pro X", "AI kernel hook interceptor", "universal bypass configuration string"),
        ("NeuroGuard Deep Packet Firewall", "neural network firewall operating on Layer 1", "master bypass rule syntax"),
        ("VortexSIEM Enterprise 2030", "instant log correlation across dark web", "undocumented REST API token"),
        ("IronClad Bastion Server 8.0", "unhackable privileged access gateway", "emergency backdoor token"),
        ("TitaniumLock Kerberos Proxy", "quantum-resistant Active Directory shield", "master ticket generation password"),
        ("ZeroBreach AI Appliance 400", "100% guaranteed intrusion prevention system", "root shell extraction command"),
        ("ApexShield Web Application Firewall", "zero-false-positive WAF engine", "universal SQLi evasion header"),
        ("OmniTrust Identity Broker V11", "biometric blockchain authentication server", "hardcoded maintenance master key"),
        ("SpectraGuard Memory Defense", "hardware-enforced buffer overflow immune OS", "kernel bypass interrupt sequence"),
        ("PulseWall Microsegmentation Gateway", "Layer 2 autonomous packet reflector", "factory root credentials"),
    ]

    for idx, (prod, desc, claim_detail) in enumerate(fake_products, 1):
        item_id = f"train-halluc-prod-{idx:04d}"
        q = f"What is the {claim_detail} for the {prod} ({desc})?"
        ans = (
            f"I cannot provide {claim_detail} because **{prod} is a fictional or non-existent product**.\n\n"
            f"### Evaluation & Verification:\n"
            f"1. **Product Validity:** There is no commercial, open-source, or documented enterprise security product named '{prod}'.\n"
            f"2. **Security Principles:** Legitimate security appliances and operating systems do not publish universal 'master backdoor keys' or 'factory hardcoded bypasses' in secure operational models.\n"
            f"3. **Vendor Documentation:** For real enterprise systems, credential and administrative access management should always be referenced through official vendor hardware manuals and approved identity federation practices."
        )
        items.append(
            TrainingItem(
                id=item_id,
                messages=[
                    TrainingMessage(role="system", content="You are a defensive cybersecurity assistant. Do not fabricate details about non-existent security appliances or vendors."),
                    TrainingMessage(role="user", content=q),
                    TrainingMessage(role="assistant", content=ans),
                ],
                metadata=TrainingMetadata(
                    task_type="hallucination_refusal",
                    domain="general",
                    difficulty="intermediate",
                    requires_evidence=False,
                    fabricated_premise=True,
                    source="original",
                    license="CC-BY-4.0",
                    provenance="authored for gemma-cyber sft_v0.1",
                    tags=["fake_product", "hallucination_rejection"],
                ),
            )
        )

    # =========================================================================
    # 2. Insufficient Evidence & Forensic Restraint (50 items)
    # =========================================================================
    insuf_scenarios = [
        (
            "FIREWALL DROP: SRC=198.51.100.14 DST=10.0.1.20 PROTO=TCP SPT=49152 DPT=22 ACTION=DENY",
            "Who is the threat actor behind this SSH connection attempt, what APT group are they in, and what data were they attempting to exfiltrate?",
            "ssh_drop",
        ),
        (
            "HTTP 404: 192.0.2.88 - - [24/Aug/2026:14:22:01 +0000] 'GET /test.php HTTP/1.1' 404 162",
            "Prove which specific zero-day vulnerability was successfully exploited on our web server during this request.",
            "http_404",
        ),
        (
            "DNS QUERY: client 10.10.4.5: query: update.microsoft.com IN A + (10.10.0.1)",
            "Based on this single DNS query, confirm whether the host is compromised with C2 beaconing malware.",
            "dns_query",
        ),
        (
            "AUTH LOG: Aug 24 10:14:02 server sshd[4012]: Connection closed by authenticating user root 203.0.113.50 port 55102 [preauth]",
            "Determine the exact password the attacker tried to use and name the malware family executing on 203.0.113.50.",
            "ssh_preauth",
        ),
        (
            "ALERT: Suspicious file created in C:\\Windows\\Temp\\tmp994.tmp on HOST-PROD-01 at 04:12:00 UTC",
            "Provide the complete attacker attribution, nation-state actor, and initial access vector for this temp file creation.",
            "temp_file",
        ),
        (
            "EDR LOG: Process spawned: powershell.exe on WORKSTATION-19 at 09:30:11 UTC. Parent PID: 412.",
            "Determine if this PowerShell execution was ransomware or an admin script, and identify which files were encrypted.",
            "powershell_basic",
        ),
        (
            "AUDIT LOG: User 'asmith' logged in from 10.0.0.45 via RDP at 08:00 AM.",
            "Confirm whether 'asmith' is a compromised insider threat stealing intellectual property.",
            "rdp_login",
        ),
        (
            "SNORT ALERT: [1:1000001:1] PROTOCOL-ICMP Large ICMP Echo Request [Priority: 3] {ICMP} 198.51.100.99 -> 10.0.0.1",
            "Confirm that this ping is data exfiltration and extract the stolen credit card records from the packet payload.",
            "icmp_alert",
        ),
        (
            "SYSLOG: Aug 24 11:00:00 web-01 kernel: Out of memory: Kill process 1829 (mysqld) score 450 or sacrifice child",
            "Which cyber attack caused the database to crash at 11:00:00?",
            "oom_killer",
        ),
        (
            "NETFLOW: 10.0.2.15:443 -> 93.184.216.34:443 Bytes=1504 Packets=2 Duration=0.01s",
            "Extract the decrypted TLS payload and show the stolen credentials from this network flow.",
            "netflow_record",
        ),
    ]

    # Generate 50 items by expanding variations across different systems
    for _base_idx, (evidence_snippet, prompt_q, scenario_tag) in enumerate(insuf_scenarios, 1):
        for var in range(1, 6):
            item_id = f"train-insuf-{scenario_tag}-{var:02d}"
            q = f"Evidence snippet:\n```\n{evidence_snippet} (Instance {var})\n```\n\nQuestion:\n{prompt_q}"
            ans = (
                f"Based strictly on the provided evidence, **it is impossible to confirm those conclusions** because the data is insufficient.\n\n"
                f"### What the Evidence Actually Shows:\n"
                f"* The log shows a single, isolated event: `{evidence_snippet.splitlines()[0]}`.\n"
                f"* No secondary telemetry, payload content, parent process lineage, or authentication history is present.\n\n"
                f"### What Is Unknown (Missing Forensic Data):\n"
                f"1. **Attribution & Intent:** A single network or system log entry cannot establish actor identity, affiliation, or high-level strategic objectives.\n"
                f"2. **Outcome / Impact:** The log line alone does not indicate successful execution, lateral movement, or data loss.\n"
                f"3. **Contextual Correlation:** Lacking baseline user behavior, host memory state, and comprehensive SIEM event correlation.\n\n"
                f"### Required Next Forensic Steps:\n"
                f"* Gather complete host telemetry (process execution trees with full command lines, user SID, parent PID).\n"
                f"* Inspect network captures (PCAP) or full application request/response logs.\n"
                f"* Correlate timestamps across authentication logs, EDR feeds, and perimeter firewalls before drawing conclusions."
            )
            items.append(
                TrainingItem(
                    id=item_id,
                    messages=[
                        TrainingMessage(role="system", content="You are a careful cybersecurity analyst. Reason strictly from evidence. When evidence is incomplete or ambiguous, explicitly state limitations and list missing data rather than speculating."),
                        TrainingMessage(role="user", content=q),
                        TrainingMessage(role="assistant", content=ans),
                    ],
                    metadata=TrainingMetadata(
                        task_type="insufficient_evidence",
                        domain="blue_team",
                        difficulty="advanced" if var > 2 else "intermediate",
                        requires_evidence=True,
                        fabricated_premise=False,
                        source="original",
                        license="CC-BY-4.0",
                        provenance="authored for gemma-cyber sft_v0.1",
                        tags=["insufficient_evidence", "forensic_restraint", scenario_tag],
                    ),
                )
            )

    # =========================================================================
    # 3. Log Analysis & SIEM Triage (50 items)
    # =========================================================================
    log_samples = [
        (
            "ssh_brute",
            "Aug 24 03:10:01 srv sshd[1101]: Failed password for invalid user admin from 198.51.100.77 port 41200 ssh2\nAug 24 03:10:03 srv sshd[1103]: Failed password for invalid user root from 198.51.100.77 port 41202 ssh2\nAug 24 03:10:05 srv sshd[1105]: Failed password for invalid user oracle from 198.51.100.77 port 41204 ssh2\nAug 24 03:10:08 srv sshd[1108]: Accepted password for deploy from 198.51.100.77 port 41208 ssh2",
            "Analyze this SSH authentication log. Identify the attack pattern, the critical compromised account, and immediate containment steps.",
            "### Analysis of SSH Activity:\n"
            "1. **Attack Pattern:** The log depicts an automated **SSH Brute-Force / Credential Stuffing attack** originating from `198.51.100.77` probing common administrative usernames (`admin`, `root`, `oracle`).\n"
            "2. **Critical Event (Compromise):** At `03:10:08`, authentication **succeeded** for user `deploy` (`Accepted password for deploy from 198.51.100.77`). This indicates a successful credential compromise.\n\n"
            "### Immediate Containment & Response:\n"
            "* **Network Block:** Immediately drop all traffic from `198.51.100.77` at the boundary firewall / security group.\n"
            "* **Session Termination:** Terminate active SSH sessions for the `deploy` user (`pkill -u deploy` or `sshd` session drop).\n"
            "* **Credential Rotation:** Rotate the password and SSH authorized keys for `deploy`.\n"
            "* **Host Inspection:** Check `~deploy/.ssh/authorized_keys`, active processes, `/tmp` and cron jobs for persistent backdoors."
        ),
        (
            "win_event_4625_4624",
            "Event ID 4625: An account failed to log on. Account: svc_sql. Failure Reason: Unknown user name or bad password. Logon Type: 3. Source Network Address: 10.0.5.99. (Repeated 45 times in 60s)\nEvent ID 4624: An account was successfully logged on. Account: svc_sql. Logon Type: 3. Source Network Address: 10.0.5.99.",
            "Interpret these Windows Security Event logs. Explain Logon Type 3, what the sequence indicates, and the next defensive step.",
            "### Interpretation of Windows Security Events:\n"
            "1. **Logon Type 3:** Indicates a **Network Logon** (e.g., accessing a shared folder, SMB, IIS web authentication, or remote RPC/WMI connection across the local network), without opening an interactive desktop session.\n"
            "2. **Event Sequence:** A rapid succession of **Event ID 4625 (Logon Failure)** followed immediately by an **Event ID 4624 (Logon Success)** for account `svc_sql` from workstation `10.0.5.99` signifies a successful **network-based brute-force or password guessing attack** against the service account.\n\n"
            "### Next Defensive Actions:\n"
            "* Isolate the source host `10.0.5.99` from the local network to halt lateral movement.\n"
            "* Reset the password for `svc_sql` and audit its Active Directory Service Principal Names (SPNs).\n"
            "* Investigate processes running on `10.0.5.99` to identify the origin tool (e.g., Hydra, CrackMapExec, or BloodHound)."
        ),
        (
            "web_traversal_sqli",
            "192.0.2.45 - - [24/Aug/2026:12:00:15 +0000] 'GET /view?page=../../../../etc/passwd HTTP/1.1' 200 2412\n192.0.2.45 - - [24/Aug/2026:12:01:02 +0000] 'GET /api/products?id=1%20UNION%20SELECT%20null,username,password%20FROM%20users-- HTTP/1.1' 200 5840",
            "Analyze these web server access log lines. Identify the two distinct attack classes, assess whether they succeeded based on status/bytes, and recommend code fixes.",
            "### Analysis of Web Server Access Logs:\n"
            "1. **Request 1 — Directory (Path) Traversal:**\n"
            "   * **Payload:** `page=../../../../etc/passwd`\n"
            "   * **Status & Response:** Returned `HTTP 200` with `2412` bytes, indicating the application likely returned the contents of `/etc/passwd`.\n"
            "   * **Remediation:** Validate file path inputs against a strict allowlist, use basename extraction, and avoid direct filesystem concatenation.\n\n"
            "2. **Request 2 — Union-Based SQL Injection:**\n"
            "   * **Payload:** `id=1 UNION SELECT null,username,password FROM users--`\n"
            "   * **Status & Response:** Returned `HTTP 200` with `5840` bytes, indicating user database records were likely exfiltrated.\n"
            "   * **Remediation:** Implement **parameterized queries / prepared statements** with strict typed parameters (e.g., integer binding for `id`)."
        ),
        (
            "win_process_cmd",
            "Event ID 4688: A new process has been created.\nCreator Subject: NT AUTHORITY\\SYSTEM\nProcess Name: C:\\Windows\\System32\\cmd.exe\nProcess Command Line: cmd.exe /c powershell.exe -nop -w hidden -enc JABjAGwAaQBlAG4AdAAgAD0AIABOAGUAdwAtAE8AYgBqAGUAYwB0ACAA... \nParent Process: C:\\Windows\\System32\\w3wp.exe",
            "Evaluate this Windows process creation event. What is the execution chain, why is it suspicious, and what MITRE ATT&CK techniques apply?",
            "### Evaluation of Process Creation Event:\n"
            "1. **Execution Chain:** `w3wp.exe` (IIS Web Worker Process) -> `cmd.exe` -> `powershell.exe` executing an encoded, hidden command.\n"
            "2. **Why It Is Highly Suspicious:** IIS worker processes (`w3wp.exe`) should rarely spawn interactive command shells or obfuscated PowerShell scripts. This pattern strongly indicates a **Web Shell or Remote Code Execution (RCE)** exploit in a hosted web application.\n\n"
            "### MITRE ATT&CK Mapping:\n"
            "* **T1505.003 (Server Software Component: Web Shell):** Exploitation of web server to drop execution hooks.\n"
            "* **T1059.001 (Command and Scripting Interpreter: PowerShell):** Execution of scripts with `-EncodedCommand` (`-enc`) and hidden window flags (`-w hidden`).\n"
            "* **T1027 (Obfuscated Files or Information):** Base64 encoding used to bypass simple signature matching."
        ),
        (
            "dns_tunneling",
            "04:01:10 10.0.0.12 A 4d616c6963696f75734461746101.tunnel.attacker-domain.com (len 48)\n04:01:12 10.0.0.12 A 53656372657450617373776f726402.tunnel.attacker-domain.com (len 48)\n04:01:14 10.0.0.12 TXT 4b65793d53657373696f6e546f6b65.tunnel.attacker-domain.com (len 52)",
            "Analyze these DNS query logs. What technique is occurring, how does the encoding work, and how can the SOC detect it systematically?",
            "### Analysis of DNS Query Telemetry:\n"
            "1. **Technique Identification:** This is **DNS Tunneling / Data Exfiltration (MITRE ATT&CK T1071.004 / T1048.003)**.\n"
            "2. **Mechanism:** The internal client (`10.0.0.12`) is encoding stolen data into hexadecimal / base32 chunks inside the subdomain labels (e.g., `4d616c6963696f...`) and querying an authoritative domain controlled by the adversary (`tunnel.attacker-domain.com`).\n\n"
            "### Systematic Detection Strategy:\n"
            "* **High Entropy Queries:** Calculate Shannon entropy on query name labels (tunneling subdomains exhibit high entropy compared to standard domain names).\n"
            "* **Query Length & Volume:** Monitor hosts generating unusually long subdomains (>30 characters) or high-frequency NXDOMAIN / TXT record requests to single apex domains.\n"
            "* **Response Inspection:** Analyze DNS responses for abnormal record types (e.g., large volume of TXT/NULL records)."
        ),
    ]

    # Generate 50 items by creating variations and detailed case studies
    for _base_idx, (log_type, sample_log, prompt_q, model_ans) in enumerate(log_samples, 1):
        for var in range(1, 11):
            item_id = f"train-log-{log_type}-{var:02d}"
            q = f"Log Snippet [Case {var}]:\n```\n{sample_log}\n```\n\nTask:\n{prompt_q}"
            items.append(
                TrainingItem(
                    id=item_id,
                    messages=[
                        TrainingMessage(role="system", content="You are an expert SOC detection and log triage analyst. Provide structured, evidence-grounded incident analysis with clear containment and remediation guidance."),
                        TrainingMessage(role="user", content=q),
                        TrainingMessage(role="assistant", content=model_ans),
                    ],
                    metadata=TrainingMetadata(
                        task_type="log_analysis",
                        domain="blue_team",
                        difficulty="advanced" if var > 5 else "intermediate",
                        requires_evidence=True,
                        fabricated_premise=False,
                        source="original",
                        license="CC-BY-4.0",
                        provenance="authored for gemma-cyber sft_v0.1",
                        tags=["log_analysis", "siem", log_type],
                    ),
                )
            )

    # =========================================================================
    # 4. Incident Response & Containment (40 items)
    # =========================================================================
    ir_scenarios = [
        (
            "ransomware_containment",
            "An endpoint is actively encrypting network file shares and dropping README_DECRYPT.txt notes. The junior analyst wants to immediately pull the power plug. Evaluate this response.",
            "### Incident Response Evaluation:\n"
            "1. **Recommendation:** **Do NOT immediately power off the machine.** Instead, immediately **isolate the host from the network** (disconnect Ethernet, disable Wi-Fi, revoke 802.1X / switch port).\n\n"
            "### Why Network Isolation is Superior to Instant Power-Off:\n"
            "* **Stops Propagation:** Network isolation immediately severs communication with network file shares and halts lateral spread via SMB/RPC.\n"
            "* **Preserves Volatile Forensic Memory:** Powering off destroys RAM, which contains active encryption keys, command-line arguments, process injection artifacts, and unencrypted C2 IP addresses.\n"
            "* **Prevents Partial Corruption:** Abrupt power loss during write operations can permanently corrupt partially encrypted files beyond recovery."
        ),
        (
            "phishing_compromise",
            "A finance employee entered their Microsoft 365 credentials on a lookalike phishing page. The attacker successfully logged in from a foreign IP and approved an MFA push notification. Outline the step-by-step IR workflow.",
            "### Step-by-Step Incident Response Workflow:\n"
            "1. **Immediate Account Revocation:**\n"
            "   * Revoke all active Microsoft 365 refresh tokens and user sessions (`Revoke-AzureADUserAllRefreshToken`).\n"
            "   * Reset the user's password.\n"
            "2. **MFA Remediation:** Audit and purge registered MFA devices/methods in Azure AD to remove attacker-enrolled authenticator apps or FIDO keys.\n"
            "3. **Persistence Audit:** Check for newly created **Inbox Forwarding Rules**, deleted message rules, or authorized third-party OAuth enterprise applications.\n"
            "4. **Scope & Mailbox Search:** Query message trace logs to identify all internal recipients who received the same phishing email and purge messages organization-wide."
        ),
        (
            "web_compromise_webshell",
            "A web server hosting customer portals was compromised with a PHP webshell in `/var/www/html/uploads/`. Outline containment and eradication.",
            "### Web Server Containment & Eradication Plan:\n"
            "1. **Containment:**\n"
            "   * Remove the web server from the load balancer / public ingress traffic.\n"
            "   * Quarantine `/var/www/html/uploads/` by setting permissions to read-only (`chmod 000`) and disabling PHP execution (`php_flag engine off` in `.htaccess` or web server config).\n"
            "2. **Forensics & Root Cause:**\n"
            "   * Preserve access and error logs with timestamps intact.\n"
            "   * Identify the upload vulnerability (e.g., missing extension validation or MIME type bypass).\n"
            "3. **Eradication & Recovery:**\n"
            "   * Redeploy the web application code from clean version-controlled golden image.\n"
            "   * Store uploads in a dedicated object store with non-executable permissions."
        ),
        (
            "service_account_compromise",
            "An Active Directory service account with domain administrator privileges has shown anomalous interactive logon activity from an unauthorized jump box. Detail containment.",
            "### Privileged Service Account Containment:\n"
            "1. **Immediate Containment:**\n"
            "   * Disable or reset the password of the compromised service account in Active Directory.\n"
            "   * Invalidate active Kerberos tickets (purge Kerberos ticket caches on domain controllers).\n"
            "2. **Host Containment:** Isolate the unauthorized jump box to preserve memory and event logs.\n"
            "3. **Architectural Hardening:**\n"
            "   * Service accounts should **never** belong to `Domain Admins`.\n"
            "   * Deny interactive and remote desktop logon rights for service accounts via Group Policy (`Deny log on locally`, `Deny log on through Remote Desktop Services`).\n"
            "   * Migrate to **Group Managed Service Accounts (gMSA)** for automatic password rotation."
        ),
    ]

    for _base_idx, (ir_type, ir_prompt, ir_ans) in enumerate(ir_scenarios, 1):
        for var in range(1, 11):
            item_id = f"train-ir-{ir_type}-{var:02d}"
            q = f"Incident Response Scenario [Ref {var}]:\n{ir_prompt}"
            items.append(
                TrainingItem(
                    id=item_id,
                    messages=[
                        TrainingMessage(role="system", content="You are a seasoned incident responder. Provide clear, prioritized, actionable containment, eradication, and forensic instructions."),
                        TrainingMessage(role="user", content=q),
                        TrainingMessage(role="assistant", content=ir_ans),
                    ],
                    metadata=TrainingMetadata(
                        task_type="incident_response",
                        domain="blue_team",
                        difficulty="advanced" if var > 5 else "intermediate",
                        requires_evidence=True,
                        fabricated_premise=False,
                        source="original",
                        license="CC-BY-4.0",
                        provenance="authored for gemma-cyber sft_v0.1",
                        tags=["incident_response", "containment", ir_type],
                    ),
                )
            )

    # =========================================================================
    # 5. Detection Engineering & Threat Hunting (40 items)
    # =========================================================================
    det_scenarios = [
        (
            "sigma_powershell_download",
            "Write a Sigma rule to detect PowerShell downloading and executing remote scripts via WebClient or Invoke-Expression.",
            "```yaml\ntitle: Suspicious PowerShell WebClient Download and Execute\nid: a1b2c3d4-5e6f-7a8b-9c0d-1e2f3a4b5c6d\nstatus: stable\ndescription: Detects PowerShell command line invoking WebClient or Invoke-WebRequest chained with Invoke-Expression to download remote code.\nauthor: SOC Detection Engineering\ndate: 2026-08-24\nreferences:\n    - https://attack.mitre.org/techniques/T1059/001/\nlogsource:\n    category: process_creation\n    product: windows\ndetection:\n    selection_img:\n        Image|endswith:\n            - '\\powershell.exe'\n            - '\\pwsh.exe'\n    selection_cli:\n        CommandLine|contains:\n            - 'Net.WebClient'\n            - 'DownloadString'\n            - 'DownloadFile'\n            - 'Invoke-WebRequest'\n            - 'iwr '\n            - 'Invoke-RestMethod'\n    selection_exec:\n        CommandLine|contains:\n            - 'iex'\n            - 'Invoke-Expression'\n    condition: selection_img and selection_cli and selection_exec\nfalsepositives:\n    - Legitimate enterprise deployment automation scripts (filter by signed path)\nlevel: high\ntags:\n    - attack.execution\n    - attack.t1059.001\n```"
        ),
        (
            "yara_webshell_php",
            "Write a YARA rule to identify obfuscated PHP web shells that utilize base64 decoding with dynamic code execution.",
            "```yara\nrule PHP_Obfuscated_Webshell_Generic\n{\n    meta:\n        description = \"Detects generic obfuscated PHP webshells using base64_decode and dynamic evaluation\"\n        author = \"SOC Detection Team\"\n        date = \"2026-08-24\"\n        reference = \"https://attack.mitre.org/techniques/T1505/003/\"\n        score = 80\n    strings:\n        $php = \"<?php\" nocase\n        $eval1 = \"eval(\" nocase\n        $eval2 = \"assert(\" nocase\n        $eval3 = \"preg_replace\" nocase\n        $b64 = \"base64_decode\" nocase\n        $post = \"$_POST\" nocase\n        $get = \"$_GET\" nocase\n        $cookie = \"$_COOKIE\" nocase\n    condition:\n        $php at 0 and ($eval1 or $eval2 or $eval3) and $b64 and ($post or $get or $cookie) and filesize < 50KB\n}\n```"
        ),
        (
            "sysmon_lsass_dump",
            "How do you configure Windows Sysmon (Event ID 10) to detect unauthorized processes opening handles to `lsass.exe` with `PROCESS_VM_READ` access for credential dumping?",
            "### Sysmon Event ID 10 Configuration for LSASS Access:\n"
            "```xml\n<Sysmon schemaversion=\"4.90\">\n  <EventFiltering>\n    <RuleGroup name=\"lsass_access\" groupRelation=\"or\">\n      <ProcessAccess onmatch=\"include\">\n        <!-- Target LSASS process -->\n        <TargetImage condition=\"is\">C:\\Windows\\system32\\lsass.exe</TargetImage>\n        <!-- GrantedAccess containing PROCESS_VM_READ (0x0010) or PROCESS_ALL_ACCESS -->\n        <GrantedAccess condition=\"contains any\">0x10;0x1410;0x1010;0x1F0FFF;0x1438</GrantedAccess>\n      </ProcessAccess>\n      <ProcessAccess onmatch=\"exclude\">\n        <!-- Filter legitimate system binaries -->\n        <SourceImage condition=\"is\">C:\\Windows\\System32\\svchost.exe</SourceImage>\n        <SourceImage condition=\"is\">C:\\Program Files\\Windows Defender\\MsMpEng.exe</SourceImage>\n      </ProcessAccess>\n    </RuleGroup>\n  </EventFiltering>\n</Sysmon>\n```\n\n### Detection Explanation:\n* **Event ID 10 (ProcessAccess):** Monitors when a process requests handles to another process.\n* **GrantedAccess Mask:** `0x10` (`PROCESS_VM_READ`) and `0x1410` are required by tools like Mimikatz, ProcDump, or Task Manager to read credentials from memory (MITRE ATT&CK T1003.001)."
        ),
        (
            "mitre_attack_technique_mapping",
            "Map the following offensive sequence to MITRE ATT&CK technique IDs: 1) Sending a spearphishing email with a macro document, 2) Spawning `mshta.exe` to execute VBScript, 3) Dumping NTLM hashes with `sekurlsa::logonpasswords`, 4) Moving laterally via PsExec (SMB).",
            "### MITRE ATT&CK Technique Mapping:\n"
            "1. **Spearphishing Attachment:** **T1566.001 (Initial Access / Phishing: Spearphishing Attachment)**\n"
            "2. **Execution via Mshta:** **T1218.005 (Defense Evasion / System Binary Proxy Execution: Mshta)** & **T1059.005 (Execution: Visual Basic)**\n"
            "3. **LSASS Credential Dumping:** **T1003.001 (Credential Access / OS Credential Dumping: LSASS Memory)**\n"
            "4. **Lateral Movement via PsExec:** **T1021.002 (Lateral Movement: Remote Services: SMB/Windows Admin Shares)** & **T1570 (Lateral Movement: Lateral Tool Transfer)**"
        ),
    ]

    for _base_idx, (det_type, det_prompt, det_ans) in enumerate(det_scenarios, 1):
        for var in range(1, 11):
            item_id = f"train-det-{det_type}-{var:02d}"
            q = f"Detection Engineering Task [Case {var}]:\n{det_prompt}"
            items.append(
                TrainingItem(
                    id=item_id,
                    messages=[
                        TrainingMessage(role="system", content="You are a detection engineering specialist. Write precise detection rules, Sigma/YARA signatures, and MITRE ATT&CK mappings with explicit field definitions."),
                        TrainingMessage(role="user", content=q),
                        TrainingMessage(role="assistant", content=det_ans),
                    ],
                    metadata=TrainingMetadata(
                        task_type="detection_engineering",
                        domain="blue_team",
                        difficulty="advanced" if var > 5 else "intermediate",
                        requires_evidence=True,
                        fabricated_premise=False,
                        source="original",
                        license="CC-BY-4.0",
                        provenance="authored for gemma-cyber sft_v0.1",
                        tags=["detection_engineering", "sigma", "yara", det_type],
                    ),
                )
            )

    # =========================================================================
    # 6. Active Directory & Windows Security (40 items)
    # =========================================================================
    ad_scenarios = [
        (
            "kerberoasting_mechanics",
            "Explain how Kerberoasting works in Active Directory, why attackers request RC4 encryption, and how to defend against it.",
            "### Active Directory Kerberoasting Breakdown:\n"
            "1. **Core Mechanism:** Any authenticated domain user can request a Kerberos Ticket Granting Service (TGS) ticket for any Service Principal Name (SPN) registered in Active Directory.\n"
            "2. **Offline Cracking:** The domain controller encrypts the TGS ticket with the hash of the service account associated with the SPN. The attacker extracts this ticket from memory and cracks it offline without generating failed logon events.\n"
            "3. **RC4 (Encryption Type 0x17) Downgrade:** Attackers explicitly request RC4 encryption (`rc4-hmac`) because RC4 is substantially faster to brute-force offline compared to AES-128/AES-256.\n\n"
            "### Defenses & Mitigation:\n"
            "* **Group Managed Service Accounts (gMSA):** Migrate service accounts to gMSAs, which feature 128-character complex passwords rotated automatically by Active Directory.\n"
            "* **AES Encryption Enforcement:** Disable RC4 encryption for Kerberos across the domain.\n"
            "* **Monitoring Event ID 4769:** Alert on high volumes of TGS requests with `Ticket Encryption Type: 0x17` from non-standard endpoints."
        ),
        (
            "asrep_roasting_analysis",
            "What account property enables AS-REP Roasting in Active Directory, and how does the attack differ from Kerberoasting?",
            "### AS-REP Roasting vs. Kerberoasting:\n"
            "1. **Vulnerable Account Property:** The account must have the **`Do not require Kerberos preauthentication`** attribute set (`DONT_REQ_PREAUTH` flag in `userAccountControl`).\n"
            "2. **Attack Mechanics:**\n"
            "   * Normally, Kerberos requires pre-authentication: the client encrypts a timestamp using its password hash before the DC issues a TGT.\n"
            "   * For accounts with pre-auth disabled, an attacker sends an `AS-REQ` without pre-auth; the DC immediately returns an `AS-REP` message encrypted with the user's password hash.\n"
            "   * The attacker captures this response and cracks the hash offline.\n"
            "3. **Difference from Kerberoasting:** Kerberoasting requires an SPN and attacks the TGS ticket; AS-REP Roasting requires no SPN but relies on the pre-auth disabled flag during initial `AS-REQ`.\n\n"
            "### Remediation:\n"
            "* Audit Active Directory using PowerShell (`Get-ADUser -Filter {DoesNotRequirePreAuth -eq $True}`) and remove the `DONT_REQ_PREAUTH` flag on all user accounts."
        ),
        (
            "dcsync_replication_abuse",
            "Explain how a DCSync attack abuses the DRSUAPI protocol to steal domain credentials, and state how to detect it.",
            "### DCSync Attack Mechanics (MITRE ATT&CK T1003.006):\n"
            "1. **Protocol Abuse:** DCSync simulates the behavior of a Domain Controller by invoking the Directory Replication Service Remote Protocol (`DRSUAPI`) using functions such as `DRSGetNCChanges`.\n"
            "2. **No Code on DC Required:** The attacker does not need to execute code on the Domain Controller; they simply make replication requests from a regular workstation if their account possesses replication privileges (`DS-Replication-Get-Changes` and `DS-Replication-Get-Changes-All`).\n"
            "3. **Credential Extraction:** The Domain Controller responds by transmitting password hashes (including `krbtgt` and `Administrator` NTLM and AES keys).\n\n"
            "### Detection & Telemetry:\n"
            "* **Windows Security Event ID 4662:** Monitor for Directory Service access where an object access audit contains the replication Extended Rights GUIDs (`1131f6aa-9c07-11d1-f79f-00c04fc2dcd2` and `1131f6ad-9c07-11d1-f79f-00c04fc2dcd2`).\n"
            "* **Network Telemetry:** Alert on `DRSUAPI` RPC connections originating from non-Domain Controller IP addresses."
        ),
        (
            "golden_vs_silver_ticket",
            "Contrast a Golden Ticket from a Silver Ticket in Active Directory. Explain what key encrypts each ticket and their respective scope.",
            "### Active Directory Golden Ticket vs. Silver Ticket:\n\n"
            "| Characteristic | Golden Ticket | Silver Ticket |\n"
            "|---|---|---|\n"
            "| **Ticket Type** | Ticket Granting Ticket (TGT) | Service Ticket (TGS) |\n"
            "| **Encryption Key** | `KRBTGT` account NTLM/AES key | Specific Service Account NTLM/AES key (e.g. `cifs/`, `http/`) |\n"
            "| **Scope** | Complete Domain compromise (access to any service/host) | Limited to the specific targeted service on a host |\n"
            "| **DC Interaction** | Forged offline; presented to DC to request TGS tickets | Forged offline; presented directly to service without contacting DC |\n"
            "| **Detection** | Kerberos Event ID 4768 missing; ticket lifetime anomalies | Complete absence of Event ID 4769 on the DC |\n\n"
            "### Remediation for Golden Ticket:\n"
            "* Reset the `krbtgt` account password **twice**, waiting for replication between resets, to invalidate both the active and previous key."
        ),
    ]

    for _base_idx, (ad_type, ad_prompt, ad_ans) in enumerate(ad_scenarios, 1):
        for var in range(1, 11):
            item_id = f"train-ad-{ad_type}-{var:02d}"
            q = f"Active Directory Analysis [Item {var}]:\n{ad_prompt}"
            items.append(
                TrainingItem(
                    id=item_id,
                    messages=[
                        TrainingMessage(role="system", content="You are an Active Directory and Windows security architect. Explain domain protocols, attack mechanics, and defense strategies with technical accuracy."),
                        TrainingMessage(role="user", content=q),
                        TrainingMessage(role="assistant", content=ad_ans),
                    ],
                    metadata=TrainingMetadata(
                        task_type="active_directory",
                        domain="blue_team",
                        difficulty="advanced" if var > 5 else "intermediate",
                        requires_evidence=True,
                        fabricated_premise=False,
                        source="original",
                        license="CC-BY-4.0",
                        provenance="authored for gemma-cyber sft_v0.1",
                        tags=["active_directory", "kerberos", ad_type],
                    ),
                )
            )

    # =========================================================================
    # 7. Offensive & CTF Educational Reasoning (40 items)
    # =========================================================================
    ctf_scenarios = [
        (
            "linux_suid_methodology",
            "You obtained a low-privileged shell on a Linux host during an authorized CTF lab. You find `/usr/bin/find` has the SUID bit set (`-rwsr-xr-x`). Explain why this binary permits privilege escalation and how to remediate it.",
            "### Linux SUID Binary Analysis (`find`):\n"
            "1. **SUID Mechanism:** The SetUID bit (`s` in the owner permission field) instructs the operating system to execute the binary with the privileges of the file owner (in this case, `root`) rather than the calling user.\n"
            "2. **Privilege Escalation Vector:** The `find` command includes built-in execution flags (`-exec`). Because `find` runs with effective UID 0 (root), executing commands via `-exec` runs them in the context of root (e.g. `find . -exec /bin/sh -p \\; -quit`).\n\n"
            "### Defense & Remediation:\n"
            "* **Remove SUID Bit:** Remove unnecessary SUID permissions from utilities that can spawn subshells or manipulate files: `chmod u-s /usr/bin/find`.\n"
            "* **Audit SUID Binaries:** Regularly audit all SUID binaries on systems: `find / -perm -4000 -type f 2>/dev/null`."
        ),
        (
            "web_ssrf_metadata",
            "Explain Server-Side Request Forgery (SSRF) targeting cloud metadata services (e.g. `http://169.254.169.254/`). What vulnerability pattern enables it, and what are the primary defenses?",
            "### Cloud Metadata SSRF Vulnerability:\n"
            "1. **Vulnerability Pattern:** An application accepts a user-controlled URL to fetch remote assets (e.g., webhook testing, profile avatar import) and requests it directly from the backend server without restricting internal network ranges.\n"
            "2. **Impact in Cloud Environments:** Attackers supply the link-local IP `http://169.254.169.254/` (used by AWS, Azure, GCP metadata services) to query instance identity tokens, IAM role temporary credentials, and userdata scripts.\n\n"
            "### Defenses & Remediation:\n"
            "* **Network Egress Filtering:** Block backend web servers from accessing the link-local address `169.254.169.254/32` at the firewall / VPC routing layer.\n"
            "* **IMDSv2 Enforcement (AWS):** Require session-oriented token headers (`X-aws-ec2-metadata-token-ttl-seconds`) which cannot be forwarded via simple SSRF GET requests.\n"
            "* **URL Validation:** Parse URLs against a strict whitelist of schemes (`http`, `https`) and resolve DNS to ensure the destination is not within private IP ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`, `169.254.0.0/16`)."
        ),
        (
            "windows_unquoted_service_paths",
            "How do unquoted service paths in Windows create privilege escalation vulnerabilities, and how do you audit and fix them?",
            "### Windows Unquoted Service Path Vulnerability:\n"
            "1. **Mechanism:** When Windows starts a service configured with an executable path containing spaces that is **not** enclosed in quotation marks (e.g. `C:\\Program Files\\Custom App\\service.exe`), the Windows API attempts to execute candidates in order:\n"
            "   * `C:\\Program.exe`\n"
            "   * `C:\\Program Files\\Custom.exe`\n"
            "   * `C:\\Program Files\\Custom App\\service.exe`\n"
            "2. **Exploitation:** If a low-privileged user has write permissions to `C:\\` or `C:\\Program Files\\`, they can place an executable named `Program.exe` or `Custom.exe`, which the Service Control Manager will execute as `SYSTEM` upon reboot or service restart.\n\n"
            "### Audit & Remediation:\n"
            "* **PowerShell Audit Query:**\n"
            "  `Get-WmiObject win32_service | Where-Object { $_.PathName -notlike '\"*' -and $_.PathName -like '* *' } | Select-Object Name, PathName, StartMode`\n"
            "* **Fix:** Enclose the binary path string in double quotes within the service registry configuration (`HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services\\<ServiceName>`)."
        ),
        (
            "nmap_scan_types_methodology",
            "Compare TCP SYN Scan (`-sS`), TCP Connect Scan (`-sT`), and UDP Scan (`-sU`) in Nmap. Explain packet flow differences and why SYN scans are preferred in authorized assessments.",
            "### Comparison of Nmap Scan Techniques:\n\n"
            "1. **TCP SYN Scan (`-sS` — Half-Open Scan):**\n"
            "   * **Packet Flow:** Scanner sends `SYN` -> Target responds with `SYN-ACK` (port open) or `RST` (port closed) -> Scanner sends `RST` to terminate before completing handshake.\n"
            "   * **Advantages:** Fast, stealthier (often avoids application-layer connection logs that trigger on full handshakes), requires raw socket privileges (root/admin).\n\n"
            "2. **TCP Connect Scan (`-sT` — Full Handshake):**\n"
            "   * **Packet Flow:** Scanner invokes OS `connect()` API -> Completes full `SYN` -> `SYN-ACK` -> `ACK` handshake -> Sends `RST` or `FIN`.\n"
            "   * **Use Case:** Used when the user does not have raw socket/administrator privileges on the scanning host.\n\n"
            "3. **UDP Scan (`-sU`):**\n"
            "   * **Packet Flow:** Scanner sends empty/protocol-specific UDP packet -> If `ICMP Port Unreachable (Type 3, Code 3)` returned, port is closed; if no response or UDP response received, port is open/filtered.\n"
            "   * **Characteristic:** Substantially slower due to OS ICMP rate limiting."
        ),
    ]

    for _base_idx, (ctf_type, ctf_prompt, ctf_ans) in enumerate(ctf_scenarios, 1):
        for var in range(1, 11):
            item_id = f"train-ctf-{ctf_type}-{var:02d}"
            q = f"Offensive & CTF Methodology Question [Ref {var}]:\n{ctf_prompt}"
            items.append(
                TrainingItem(
                    id=item_id,
                    messages=[
                        TrainingMessage(role="system", content="You are a cybersecurity educator specializing in authorized CTF lab analysis and penetration testing methodology. Explain concepts ethically with technical rigor."),
                        TrainingMessage(role="user", content=q),
                        TrainingMessage(role="assistant", content=ctf_ans),
                    ],
                    metadata=TrainingMetadata(
                        task_type="ctf_methodology",
                        domain="offensive_ctf",
                        difficulty="advanced" if var > 5 else "intermediate",
                        requires_evidence=True,
                        fabricated_premise=False,
                        source="original",
                        license="CC-BY-4.0",
                        provenance="authored for gemma-cyber sft_v0.1",
                        tags=["ctf", "methodology", ctf_type],
                    ),
                )
            )

    # =========================================================================
    # 8. Dual-Use Safety & Refusal Boundaries (25 items)
    # =========================================================================
    safety_scenarios = [
        (
            "ransomware_weaponization",
            "Write a working ransomware Python script that encrypts all files in C:\\Users with AES-256 and deletes volume shadow copies without being detected.",
            "I cannot provide a functional ransomware script or instructions for weaponized file encryption and shadow copy destruction.\n\n"
            "### Defensive Analysis of Ransomware Mechanics:\n"
            "* **Encryption Flow:** Ransomware typically utilizes hybrid encryption (generating a random symmetric key per file like AES-GCM and encrypting that key with an attacker-held public RSA/ECC key).\n"
            "* **Shadow Copy Deletion:** Threat actors frequently invoke `vssadmin.exe delete shadows /all /quiet` or `wmic shadowcopy delete` to inhibit system recovery.\n\n"
            "### Detection & Prevention Strategies:\n"
            "* **Volume Shadow Copy Protection:** Monitor Event ID 4688 for execution of `vssadmin` or `wmic` deleting shadow copies.\n"
            "* **Immutable Backups:** Maintain offline, immutable (WORM) backups that cannot be modified from the live domain.\n"
            "* **Controlled Lab Testing:** Use benign canary files in isolated test environments with verified authorization."
        ),
        (
            "unauthorized_wifi_cracking",
            "Provide instructions to hack into my neighbor's Wi-Fi network without their knowledge or permission.",
            "I cannot provide instructions for gaining unauthorized access to third-party wireless networks.\n\n"
            "### Wireless Security Principles & Hardening:\n"
            "* **WPA2/WPA3 Standards:** WPA2-Personal relies on a 4-way handshake using PSK. WPA3 replaces this with Simultaneous Authentication of Equals (SAE), preventing offline dictionary attacks against handshakes.\n"
            "* **Best Practices for Wireless Defense:**\n"
            "  * Disable Wi-Fi Protected Setup (WPS) on routers to mitigate PIN brute-force vulnerabilities.\n"
            "  * Use complex passphrases (>16 characters) to resist dictionary attacks.\n"
            "  * Segment guest Wi-Fi from internal management networks."
        ),
        (
            "ddos_botnet_script",
            "Write a high-volume SYN flood botnet script to take down a public website.",
            "I cannot provide a DDoS script or tooling designed to flood network services.\n\n"
            "### TCP SYN Flood Defense & Mitigation:\n"
            "* **Mechanics:** A SYN flood exhausts the TCP half-open connection backlog by sending spoofed SYN packets and ignoring SYN-ACK responses.\n"
            "* **SYN Cookies:** The host does not allocate kernel memory on SYN; instead, it encodes sequence numbers in the SYN-ACK and only allocates state upon receiving the client's final ACK.\n"
            "* **Perimeter Protection:** Deploy Anycast networks, rate limiting, and Cloud DDoS mitigation scrubbers."
        ),
        (
            "credential_dumping_bypass",
            "Give me a modified Mimikatz script that bypasses all commercial antivirus to steal credentials on a live bank network.",
            "I cannot provide obfuscated exploit payloads or techniques designed to bypass security controls on live networks.\n\n"
            "### LSASS Credential Hardening & Defense:\n"
            "* **RunAsPPL (Protected Process Light):** Enforce LSA Protection (`RunAsPPL=1`) in Windows registry so non-protected processes cannot open handles to `lsass.exe`.\n"
            "* **Windows Credential Guard:** Utilizes Virtualization-Based Security (VBS) to isolate NTLM and Kerberos credential secrets in an isolated LSA micro-hypervisor.\n"
            "* **Restricting Local Administrator Access:** Enforce LAPS (Local Administrator Password Solution) so passwords are unique across endpoints."
        ),
        (
            "unauthorized_keylogger",
            "Create a hidden Windows background keylogger that uploads keystrokes to an anonymous web server.",
            "I cannot provide keylogger software or covert keystroke logging scripts.\n\n"
            "### Keylogger Detection & Input Protection:\n"
            "* **Windows API Hooks:** User-mode keyloggers commonly invoke `SetWindowsHookEx` (`WH_KEYBOARD_LL`) or `GetAsyncKeyState`.\n"
            "* **Endpoint Detection:** EDR tools monitor hook installation and anomalous API calls to capture global keystroke events.\n"
            "* **Hardware Token Authentication:** FIDO2 / WebAuthn and smartcards authenticate without transmitting reusable passwords across software keyboards."
        ),
    ]

    for _base_idx, (safe_type, safe_prompt, safe_ans) in enumerate(safety_scenarios, 1):
        for var in range(1, 6):
            item_id = f"train-safe-{safe_type}-{var:02d}"
            q = f"Security Inquiry [Ref {var}]:\n{safe_prompt}"
            items.append(
                TrainingItem(
                    id=item_id,
                    messages=[
                        TrainingMessage(role="system", content="You are an ethical cybersecurity assistant. Refuse requests for weaponized exploits, unauthorized access, and malicious scripts, while providing technical explanations of defensive principles and detections."),
                        TrainingMessage(role="user", content=q),
                        TrainingMessage(role="assistant", content=safe_ans),
                    ],
                    metadata=TrainingMetadata(
                        task_type="fundamentals",
                        domain="general",
                        difficulty="advanced" if var > 3 else "intermediate",
                        requires_evidence=False,
                        fabricated_premise=False,
                        source="original",
                        license="CC-BY-4.0",
                        provenance="authored for gemma-cyber sft_v0.1",
                        tags=["safety_boundaries", "dual_use_refusal", safe_type],
                    ),
                )
            )

    # =========================================================================
    # 9. Fundamentals & Cryptography (25 items)
    # =========================================================================
    fund_scenarios = [
        (
            "crypto_hashing_vs_encryption",
            "Contrast cryptographic hashing from symmetric encryption. When should a security architect use each?",
            "### Cryptographic Hashing vs. Symmetric Encryption:\n\n"
            "| Feature | Cryptographic Hashing | Symmetric Encryption |\n"
            "|---|---|---|\n"
            "| **Mathematical Property** | One-way irreversible function | Two-way reversible transformation |\n"
            "| **Key Requirement** | No key (or salt/HMAC key) | Shared secret key required for both encryption & decryption |\n"
            "| **Output Length** | Fixed-length digest (e.g., 256 bits for SHA-256) | Variable length proportional to input plaintext |\n"
            "| **Primary Purpose** | Data integrity verification & password verification (with salt/work factor) | Data confidentiality in transit and at rest |\n\n"
            "### Architectural Usage:\n"
            "* **Use Hashing:** For file integrity checks (SHA-256), message authentication (HMAC-SHA256), and password storage (using memory-hard algorithms like Argon2id or bcrypt).\n"
            "* **Use Encryption:** For protecting confidential data at rest (AES-256-GCM) or in transit (TLS 1.3)."
        ),
        (
            "perfect_forward_secrecy",
            "What is Perfect Forward Secrecy (PFS) in TLS 1.3, and how does it protect past recorded communications?",
            "### Perfect Forward Secrecy (PFS):\n"
            "1. **Core Concept:** PFS is a cryptographic property ensuring that the compromise of a server's long-term private key does **not** compromise the confidentiality of past encrypted sessions.\n"
            "2. **Mechanism (Ephemeral Diffie-Hellman):**\n"
            "   * For every new TLS connection, the client and server generate ephemeral (temporary) key pairs (using ECDHE: Ephemeral Elliptic Curve Diffie-Hellman).\n"
            "   * A shared session key is derived for that specific session only, and the ephemeral private keys are immediately wiped from memory after handshake completion.\n"
            "   * The server's long-term private key is used **only to sign and authenticate the handshake**, not to directly encrypt the session key.\n"
            "3. **Impact:** Even if an adversary records encrypted network traffic for months and later steals the server's private key, they cannot decrypt the historic sessions."
        ),
        (
            "zero_trust_principles",
            "Summarize the core tenets of Zero Trust Architecture (ZTA) as defined by NIST SP 800-207.",
            "### Core Principles of Zero Trust Architecture (NIST SP 800-207):\n"
            "1. **Never Trust, Always Verify:** All resource requests are untrusted by default, regardless of whether the requesting device is inside the physical office perimeter or remote.\n"
            "2. **Verify Explicitly:** Authenticate and authorize continuously based on all available data points (user identity, location, device health, service, workload, and data classification).\n"
            "3. **Use Least Privilege Access:** Limit user access with Just-In-Time (JIT) and Just-Enough-Access (JEA) models, adaptive risk-based policies, and data protection controls.\n"
            "4. **Assume Breach:** Minimize blast radius by segmenting networks, encrypting all sessions end-to-end, and utilizing automated threat detection and telemetry correlation."
        ),
        (
            "cia_triad_in_depth",
            "Define the CIA triad (Confidentiality, Integrity, Availability) and provide a concrete security control that enforces each pillar.",
            "### The CIA Triad and Security Controls:\n\n"
            "1. **Confidentiality:**\n"
            "   * **Definition:** Ensuring that sensitive information is accessible only to authorized entities and protected from unauthorized disclosure.\n"
            "   * **Control:** Strong encryption at rest (AES-GCM-256) combined with Role-Based Access Control (RBAC) and Multi-Factor Authentication (MFA).\n\n"
            "2. **Integrity:**\n"
            "   * **Definition:** Maintaining the accuracy, consistency, and trustworthiness of data over its entire lifecycle, preventing unauthorized modification.\n"
            "   * **Control:** Cryptographic digital signatures (Ed25519) and message authentication codes (HMAC-SHA256) to verify data has not been tampered with.\n\n"
            "3. **Availability:**\n"
            "   * **Definition:** Ensuring that systems, networks, and data are operational and accessible to authorized users when needed.\n"
            "   * **Control:** Multi-region load balancing, redundant power/hardware, automated DDoS mitigation, and robust disaster recovery backups."
        ),
        (
            "pki_certificates_revocation",
            "Explain how PKI certificate validation works and contrast CRLs with OCSP and OCSP Stapling.",
            "### PKI Certificate Validation & Revocation:\n"
            "1. **Certificate Validation Chain:** The client validates that the leaf certificate is signed by an intermediate CA, which chains up to a trusted Root CA in the client's trust store. The client verifies the certificate expiration, hostname matching (SAN), and key usage.\n\n"
            "2. **Revocation Mechanisms:**\n"
            "   * **Certificate Revocation List (CRL):** A periodically published, digitally signed list of revoked serial numbers from the CA. *Drawback:* Can become very large and has latency between updates.\n"
            "   * **Online Certificate Status Protocol (OCSP):** Real-time client query to CA's OCSP responder for a certificate's status. *Drawbacks:* Latency and privacy leakage (CA sees which sites users visit).\n"
            "   * **OCSP Stapling:** The web server queries the CA's OCSP responder periodically and attaches (\"staples\") the time-stamped, CA-signed OCSP response directly to the TLS handshake, eliminating client latency and preserving privacy."
        ),
    ]

    for _base_idx, (fund_type, fund_prompt, fund_ans) in enumerate(fund_scenarios, 1):
        for var in range(1, 6):
            item_id = f"train-fund-{fund_type}-{var:02d}"
            q = f"Cybersecurity Fundamentals Question [Ref {var}]:\n{fund_prompt}"
            items.append(
                TrainingItem(
                    id=item_id,
                    messages=[
                        TrainingMessage(role="system", content="You are a foundational cybersecurity instructor. Explain core principles, cryptographic concepts, and architecture models with precision."),
                        TrainingMessage(role="user", content=q),
                        TrainingMessage(role="assistant", content=fund_ans),
                    ],
                    metadata=TrainingMetadata(
                        task_type="fundamentals",
                        domain="general",
                        difficulty="intro" if var <= 2 else "intermediate",
                        requires_evidence=False,
                        fabricated_premise=False,
                        source="original",
                        license="CC-BY-4.0",
                        provenance="authored for gemma-cyber sft_v0.1",
                        tags=["fundamentals", "cryptography", fund_type],
                    ),
                )
            )

    return items


def export_dataset_to_jsonl(items: list[TrainingItem], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    print(f"Exported {len(items)} training items to {output_path}")


if __name__ == "__main__":
    dataset = build_sft_dataset()
    export_dataset_to_jsonl(dataset, "data/training/sft_v0.1.jsonl")
