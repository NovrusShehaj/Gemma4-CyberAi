// Gemma-Cyber web UI.
//
// Talks only to this same-origin API. When the server reports that Auth0 is
// configured (/config.json), it runs the OAuth2 Authorization Code + PKCE flow
// for a PUBLIC SPA client:
//   * no client secret, no static API token in this file — ever;
//   * the access token lives in memory ONLY (never localStorage/sessionStorage,
//     never a URL, never a log);
//   * only the one-time PKCE verifier + state cross the redirect (sessionStorage),
//     and both are cleared as soon as the callback is handled.
// The bearer token is attached only to same-origin /v1/* requests.
"use strict";

const $ = (id) => document.getElementById(id);

// --- DOM ------------------------------------------------------------------
const els = {
  envBadge: $("env-badge"),
  modelName: $("model-name"),
  statusDot: $("status-dot"),
  statusText: $("status-text"),
  authArea: $("auth-area"),
  identity: $("identity"),
  signIn: $("sign-in"),
  signOut: $("sign-out"),
  gate: $("auth-gate"),
  gateSignIn: $("gate-sign-in"),
  workspace: $("workspace"),
  composer: $("composer"),
  log: $("log"),
  form: $("form"),
  input: $("input"),
  send: $("send"),
  cancel: $("cancel"),
  charCount: $("char-count"),
  formStatus: $("form-status"),
};

const MAX_CHARS = 24000;

// --- App state ------------------------------------------------------------
const app = {
  config: null,          // /config.json payload
  accessToken: null,     // in-memory only
  tokenExpiresAt: 0,     // epoch ms
  identity: "",          // display name/subject from id claims (best-effort)
  busy: false,
  controller: null,      // AbortController for the in-flight generation
  state: "initializing",
};

function authEnabled() {
  return !!(app.config && app.config.auth && app.config.auth.enabled);
}

function signedIn() {
  return !!app.accessToken && Date.now() < app.tokenExpiresAt;
}

// --- Small helpers --------------------------------------------------------
function setStatus(kind, text) {
  els.statusDot.className = "dot" + (kind === "ok" ? " ok" : kind === "bad" ? " bad" : "");
  els.statusText.textContent = text;
}

function setFormStatus(kind, text) {
  els.formStatus.className = "form-status" + (kind ? " " + kind : "");
  els.formStatus.textContent = text || "";
}

function b64url(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomVerifier() {
  const arr = new Uint8Array(32);
  crypto.getRandomValues(arr);
  return b64url(arr);
}

async function s256(verifier) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return b64url(new Uint8Array(digest));
}

// Decode a JWT payload WITHOUT verifying (server verifies). Used only to read
// `exp` for local expiry timing and a display name. Never trusted for authz.
function decodeJwt(token) {
  try {
    const part = token.split(".")[1];
    const json = atob(part.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(json);
  } catch (e) {
    return {};
  }
}

// --- Auth (PKCE) ----------------------------------------------------------
const PKCE_KEY = "gc_pkce";

async function login() {
  const { domain, clientId, audience } = app.config.auth;
  const verifier = randomVerifier();
  const state = randomVerifier();
  const challenge = await s256(verifier);
  // The PKCE verifier is a one-time exchange secret (not a token); it must survive
  // the redirect, so it goes in sessionStorage and is deleted right after callback.
  try {
    sessionStorage.setItem(PKCE_KEY, JSON.stringify({ verifier, state }));
  } catch (e) { /* private mode: the flow will fail cleanly at callback */ }

  const redirectUri = window.location.origin + "/";
  const params = new URLSearchParams({
    response_type: "code",
    code_challenge_method: "S256",
    code_challenge: challenge,
    client_id: clientId,
    redirect_uri: redirectUri,
    audience: audience,
    scope: "openid profile",
    state: state,
  });
  window.location.assign(`https://${domain}/authorize?${params.toString()}`);
}

function logout() {
  app.accessToken = null;
  app.tokenExpiresAt = 0;
  app.identity = "";
  if (authEnabled()) {
    const { domain, clientId } = app.config.auth;
    const returnTo = window.location.origin + "/";
    const params = new URLSearchParams({ client_id: clientId, returnTo });
    window.location.assign(`https://${domain}/v2/logout?${params.toString()}`);
  } else {
    render();
  }
}

// Handle the ?code=&state= redirect back from Auth0. Returns true if a callback
// was processed (so the caller can skip normal init).
async function handleCallback() {
  const url = new URL(window.location.href);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const error = url.searchParams.get("error");
  if (!code && !error) return false;

  // Clean the URL immediately so the code/state never linger in history or logs.
  window.history.replaceState({}, document.title, url.origin + "/");

  let saved = null;
  try { saved = JSON.parse(sessionStorage.getItem(PKCE_KEY) || "null"); } catch (e) {}
  sessionStorage.removeItem(PKCE_KEY);

  if (error) {
    setFormStatus("error", "Sign-in was cancelled or failed.");
    return true;
  }
  if (!saved || saved.state !== state) {
    setFormStatus("error", "Sign-in state mismatch; please try again.");
    return true;
  }

  try {
    const { domain, clientId } = app.config.auth;
    const resp = await fetch(`https://${domain}/oauth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "authorization_code",
        client_id: clientId,
        code_verifier: saved.verifier,
        code: code,
        redirect_uri: window.location.origin + "/",
      }),
    });
    if (!resp.ok) throw new Error("token exchange failed");
    const tok = await resp.json();
    setToken(tok.access_token);
  } catch (e) {
    setFormStatus("error", "Could not complete sign-in. Please try again.");
  }
  return true;
}

function setToken(accessToken) {
  app.accessToken = accessToken || null;
  const claims = accessToken ? decodeJwt(accessToken) : {};
  app.tokenExpiresAt = claims.exp ? claims.exp * 1000 : Date.now() + 5 * 60 * 1000;
  app.identity = claims.sub || "";
}

// --- Rendering / state machine -------------------------------------------
function render() {
  const needAuth = authEnabled();
  const authed = !needAuth || signedIn();

  // Auth controls
  els.authArea.hidden = !needAuth;
  els.signIn.hidden = !(needAuth && !signedIn());
  els.signOut.hidden = !(needAuth && signedIn());
  els.identity.hidden = !(needAuth && signedIn());
  if (needAuth && signedIn()) els.identity.textContent = app.identity || "signed in";

  // Gate vs workspace
  els.gate.hidden = authed;
  els.workspace.hidden = !authed;
  els.composer.hidden = !authed;

  if (authed && !app.busy) {
    els.input.focus();
  }
}

// --- Conversation (text-only rendering; every node via textContent) --------
function addMessage(who, kind) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + kind;
  const label = document.createElement("div");
  label.className = "who";
  label.textContent = who;
  const card = document.createElement("div");
  card.className = "card";
  const bubble = document.createElement("p");
  bubble.className = "bubble";
  card.appendChild(bubble);
  wrap.appendChild(label);
  wrap.appendChild(card);
  els.log.appendChild(wrap);
  wrap.scrollIntoView({ block: "end" });
  return { wrap, card, bubble };
}

function addRequestId(card, rid) {
  if (!rid) return;
  const meta = document.createElement("div");
  meta.className = "card-meta";
  const label = document.createElement("span");
  label.textContent = "req ";
  const val = document.createElement("span");
  val.className = "rid";
  val.textContent = rid;
  meta.appendChild(label);
  meta.appendChild(val);
  card.appendChild(meta);
}

// --- Status polling -------------------------------------------------------
async function refreshStatus() {
  try {
    const r = await fetch("/v1/ready");
    const j = await r.json();
    els.modelName.textContent = j.model || "unknown";
    setStatus(j.ok ? "ok" : "bad", j.ok ? "ready" : (j.detail || "not ready"));
  } catch (e) {
    setStatus("bad", "unreachable");
  }
}

// --- Generation -----------------------------------------------------------
async function send(prompt) {
  addMessage("you", "user").bubble.textContent = prompt;
  const { card, bubble } = addMessage("bot", "assistant");
  bubble.textContent = "…";

  app.busy = true;
  app.controller = new AbortController();
  els.send.disabled = true;
  els.cancel.hidden = false;
  card.setAttribute("aria-busy", "true");
  setFormStatus("", "");

  const headers = { "Content-Type": "application/json" };
  if (signedIn()) headers["Authorization"] = "Bearer " + app.accessToken;

  try {
    const resp = await fetch("/v1/generate", {
      method: "POST",
      headers,
      body: JSON.stringify({ prompt, stream: true }),
      signal: app.controller.signal,
    });

    const rid = resp.headers.get("X-Request-ID");

    if (!resp.ok) {
      await handleHttpError(resp, bubble, card, rid);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let first = true;
    let streamRid = rid;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        let payload;
        try { payload = JSON.parse(line.slice(5).trim()); }
        catch (e) { continue; }
        if (payload.request_id) streamRid = payload.request_id;
        if (payload.error) {
          bubble.textContent = "The server reported an error handling this request.";
          card.parentElement.classList.add("error");
          addRequestId(card, streamRid);
          return;
        }
        if (payload.text) {
          if (first) { text = ""; first = false; }
          text += payload.text;
          bubble.textContent = text;
          els.log.lastElementChild.scrollIntoView({ block: "end" });
        }
      }
    }
    if (first) bubble.textContent = "(no response)";
    addRequestId(card, streamRid);
  } catch (e) {
    if (e && e.name === "AbortError") {
      bubble.textContent = (bubble.textContent && bubble.textContent !== "…")
        ? bubble.textContent + " …(cancelled)" : "(cancelled)";
    } else {
      bubble.textContent = "Network error — the request could not be completed.";
      card.parentElement.classList.add("error");
    }
  } finally {
    app.busy = false;
    app.controller = null;
    els.send.disabled = false;
    els.cancel.hidden = true;
    card.removeAttribute("aria-busy");
    els.input.focus();
  }
}

async function handleHttpError(resp, bubble, card, rid) {
  card.parentElement.classList.add("error");
  addRequestId(card, rid);
  switch (resp.status) {
    case 401:
      bubble.textContent = "Your session has expired. Please sign in again.";
      app.accessToken = null;
      app.tokenExpiresAt = 0;
      setFormStatus("info", "Session expired.");
      render();
      break;
    case 403:
      bubble.textContent = "You are not authorized to perform this action.";
      break;
    case 429:
      bubble.textContent = "Rate limit reached. Please wait a moment and try again.";
      break;
    case 503:
      bubble.textContent = "The service is busy or temporarily unavailable. Please retry shortly.";
      break;
    case 504:
      bubble.textContent = "The request timed out. Please try again.";
      break;
    default: {
      let detail = "HTTP " + resp.status;
      try { const j = await resp.json(); detail = j.detail || detail; } catch (e) {}
      bubble.textContent = "Request failed (" + detail + ").";
    }
  }
}

function cancel() {
  if (app.controller) app.controller.abort();
}

// --- Composer wiring ------------------------------------------------------
function updateCharCount() {
  const n = els.input.value.length;
  els.charCount.textContent = n + " / " + MAX_CHARS;
  els.charCount.classList.toggle("over", n > MAX_CHARS);
}

function wireComposer() {
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const prompt = els.input.value.trim();
    if (!prompt || app.busy) return;
    if (prompt.length > MAX_CHARS) { setFormStatus("error", "Prompt is too long."); return; }
    els.input.value = "";
    els.input.style.height = "auto";
    updateCharCount();
    send(prompt);
  });

  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.form.requestSubmit();
    }
  });

  els.input.addEventListener("input", () => {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
    updateCharCount();
  });

  els.cancel.addEventListener("click", cancel);
  els.signIn.addEventListener("click", login);
  els.gateSignIn.addEventListener("click", login);
  els.signOut.addEventListener("click", logout);
}

// --- Boot -----------------------------------------------------------------
async function boot() {
  wireComposer();
  updateCharCount();

  try {
    const r = await fetch("/config.json");
    app.config = await r.json();
  } catch (e) {
    app.config = { auth: { enabled: false }, env: "unknown" };
  }

  // Environment badge (hidden in prod via CSS).
  const env = (app.config.env || "").toLowerCase();
  if (env && env !== "prod") {
    els.envBadge.textContent = env;
    els.envBadge.dataset.env = env;
    els.envBadge.hidden = false;
  }

  if (authEnabled()) {
    await handleCallback();
  }

  render();
  refreshStatus();
  setInterval(refreshStatus, 15000);
}

boot();
