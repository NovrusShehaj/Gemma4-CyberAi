// Gemma-Cyber web UI — talks only to this same-origin API (CSP: connect-src 'self').
// No third-party code, no analytics. Streaming via SSE over fetch.
"use strict";

const $ = (id) => document.getElementById(id);
const log = $("log");
const form = $("form");
const input = $("input");
const sendBtn = $("send");

let busy = false;

function addMessage(who, cls) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + cls;
  const label = document.createElement("div");
  label.className = "who";
  label.textContent = who;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrap.appendChild(label);
  wrap.appendChild(bubble);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return { wrap, bubble };
}

function setStatus(state, text) {
  const dot = $("status-dot");
  dot.className = "dot" + (state === "ok" ? " ok" : state === "bad" ? " bad" : "");
  $("status-text").textContent = text;
}

async function refreshStatus() {
  try {
    const r = await fetch("/v1/ready");
    const j = await r.json();
    $("model-name").textContent = j.model || "unknown";
    setStatus(j.ok ? "ok" : "bad", j.ok ? "ready" : (j.detail || "not ready"));
  } catch (e) {
    setStatus("bad", "unreachable");
  }
  try {
    const r = await fetch("/v1/models");
    const j = await r.json();
    if (j.production) {
      const b = $("stage-badge");
      b.innerHTML = "production <strong>" + j.production + "</strong>";
      b.hidden = false;
    }
  } catch (e) { /* registry optional */ }
}

async function send(prompt) {
  addMessage("you", "user").bubble.textContent = prompt;
  const { wrap, bubble } = addMessage("bot", "bot");
  bubble.textContent = "…";
  busy = true;
  sendBtn.disabled = true;

  try {
    const resp = await fetch("/v1/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt, stream: true }),
    });

    if (!resp.ok) {
      let detail = "HTTP " + resp.status;
      try { const j = await resp.json(); detail = j.detail || j.error || detail; } catch (e) {}
      bubble.textContent = "Error: " + detail;
      wrap.classList.add("error");
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let first = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.error) {
          bubble.textContent = "Error: " + payload.error;
          wrap.classList.add("error");
          return;
        }
        if (payload.text) {
          if (first) { text = ""; first = false; }
          text += payload.text;
          bubble.textContent = text;
          log.scrollTop = log.scrollHeight;
        }
      }
    }
    if (first) bubble.textContent = "(no response)";
  } catch (e) {
    bubble.textContent = "Error: " + (e && e.message ? e.message : "request failed");
    wrap.classList.add("error");
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const prompt = input.value.trim();
  if (!prompt || busy) return;
  input.value = "";
  input.style.height = "auto";
  send(prompt);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
});

refreshStatus();
setInterval(refreshStatus, 15000);
