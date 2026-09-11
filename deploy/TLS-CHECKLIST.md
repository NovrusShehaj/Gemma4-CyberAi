# TLS / edge go-live checklist

The application **does not terminate TLS** and **does not set HSTS**. Those
belong at the reverse proxy. The versioned reference is
[`nginx.reference.conf`](nginx.reference.conf) (example hostnames/paths only).

Certificate source is an operator decision (Let's Encrypt vs uploaded files).
This repo does not provision certificates.

## Placeholders in `nginx.reference.conf`

Replace every item below before public exposure. None of these values are live.

| Placeholder | Lines (approx) | Replace with |
|---|---|---|
| `gemma-cyber.example.com` | `server_name` (HTTP + HTTPS server blocks) | Public hostname |
| `/etc/ssl/certs/gemma-cyber.pem` | `ssl_certificate` | Real certificate path |
| `/etc/ssl/private/gemma-cyber.key` | `ssl_certificate_key` | Real private key path |
| `# set_real_ip_from 10.0.0.0/8;` | trusted proxy CIDR | Uncomment and set the **only** hop you trust |
| `# real_ip_header X-Forwarded-For;` | with the CIDR above | Uncomment when behind a trusted LB |
| `rate=60r/m` | `limit_req_zone` | Tune to capacity; keep a pre-auth cap |
| `burst=20` | `limit_req` | Tune; this is what bounds unauthenticated 401 storms |
| `server 127.0.0.1:8000;` | `upstream` | API bind (keep loopback / private net) |
| `client_max_body_size 256k;` | HTTPS server | Keep at or below app prompt bounds |
| HSTS `max-age=63072000` | `add_header` | Enable only after HTTPS is committed for this name |

The file is intentionally named a **reference** and uses `.example.com` so it
cannot be copied to `/etc/nginx` unchanged without a conscious edit.

## Before opening the host

- [ ] `docker compose -f docker-compose.yml -f docker-compose.prod.yml config` succeeds with Auth0 env set
- [ ] API still bound to `127.0.0.1:8000`; Ollama not published
- [ ] HTTPS redirect 80 → 443; `curl -I https://<host>/health` shows 200
- [ ] HSTS only after a successful HTTPS handshake on the real name
- [ ] `limit_req` enabled (app limiter does not cover 401s)
- [ ] `set_real_ip_from` is your proxy CIDR only — not `0.0.0.0/0`
- [ ] Auth0 callback / logout / web origin URLs are the **exact** HTTPS origin
- [ ] `scripts/smoke_test.py --base-url https://<host> --token "$TOKEN" --expect-auth`
