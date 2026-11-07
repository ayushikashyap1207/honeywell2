import { useEffect, useRef, useState } from "react";

function RichText({ text }) {
  const parts = String(text || "").split(/(\*\*[^*]+\*\*)/g);
  return (
    <div className="chat-text">
      {parts.map((part, index) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={index}>{part.slice(2, -2)}</strong>
        ) : (
          <span key={index}>{part}</span>
        )
      )}
    </div>
  );
}

const DEFAULT_MESSAGES = [
  {
    role: "assistant",
    text: "I'm the SOC copilot. I can explain an alert, pull entity history, find similar incidents, map the predicted attack to MITRE ATT&CK, or summarise model evaluation and drift. Pick an alert and ask me about it.",
    citations: [],
  },
];

export default function CopilotTab({ api, context }) {
  const [status, setStatus] = useState(null);
  const [messages, setMessages] = useState(DEFAULT_MESSAGES);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (!api) return;
    fetch(`${api}/api/copilot/status`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => setStatus(data))
      .catch(() => setStatus(null));
  }, [api]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function send(text) {
    const question = (text || input).trim();
    if (!question || busy) return;
    if (!api) {
      setMessages((prev) => [
        ...prev,
        { role: "user", text: question, citations: [] },
        {
          role: "assistant",
          text: "The copilot needs the FastAPI backend. Run `make demo` or start uvicorn, then try again.",
          citations: [],
        },
      ]);
      setInput("");
      return;
    }

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: question, citations: [] }]);
    setBusy(true);

    try {
      const response = await fetch(`${api}/api/copilot/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: question,
          session_id: context?.session_id || null,
          entity_id: context?.entity_id || null,
        }),
      });
      const payload = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: payload.answer || "No answer returned.",
          citations: payload.citations || [],
          provider: payload.provider,
          toolCalls: payload.tool_calls || [],
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Copilot request failed: ${error.message}`,
          citations: [],
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  const suggestions = context?.session_id
    ? [
        "Why was this session flagged?",
        "What should I do next?",
        "Show similar incidents",
        "What is the MITRE technique?",
      ]
    : [
        "How is the model performing?",
        "Is the data drifting?",
        "What does the current alert queue look like?",
      ];

  return (
    <div className="copilot-layout">
      <div className="panel copilot-panel">
        <div className="panel-header">
          <div>
            <h2>Analyst copilot</h2>
            <p>
              Grounded in alerts, entity history, MITRE ATT&CK and model reports.
              Answers cite their evidence.
            </p>
          </div>
          <span className={`badge ${status?.configured ? "badge-ok" : "badge-info"}`}>
            {status
              ? status.configured
                ? `${status.provider}:${status.model}`
                : "deterministic demo mode"
              : "checking provider"}
          </span>
        </div>

        {context?.session_id && (
          <div className="copilot-context">
            <span>
              Investigating session <strong>{context.session_id}</strong>
              {context.entity_id ? ` · ${context.entity_id}` : ""}
            </span>
          </div>
        )}

        <div className="chat-window">
          {messages.map((message, index) => (
            <div key={index} className={`chat-message ${message.role}`}>
              <div className="chat-role">
                {message.role === "user" ? "You" : "Copilot"}
                {message.provider && (
                  <span className="chat-provider">{message.provider}</span>
                )}
              </div>
              <RichText text={message.text} />
              {!!message.citations?.length && (
                <div className="citation-row">
                  {message.citations.map((citation) => (
                    <span className="citation" key={`${citation.id}-${citation.ref}`}>
                      <strong>{citation.id}</strong> {citation.label}
                    </span>
                  ))}
                </div>
              )}
              {!!message.toolCalls?.length && (
                <div className="tool-row">
                  tools: {message.toolCalls.join(", ")}
                </div>
              )}
            </div>
          ))}
          {busy && (
            <div className="chat-message assistant">
              <div className="chat-role">Copilot</div>
              <div className="chat-text typing">retrieving evidence…</div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="suggestion-row">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion}
              className="suggestion"
              onClick={() => send(suggestion)}
              disabled={busy}
            >
              {suggestion}
            </button>
          ))}
        </div>

        <div className="chat-input-row">
          <textarea
            className="chat-input"
            rows={2}
            placeholder="Ask about an alert, entity, similar incidents, model metrics or drift..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                send();
              }
            }}
          />
          <button className="send-btn" onClick={() => send()} disabled={busy}>
            Send
          </button>
        </div>
        <p className="panel-footnote">
          {status?.configured
            ? "LLM-generated answers are constrained to the evidence packet and must cite sources."
            : "No LLM is configured. The same evidence-grounded answer is generated deterministically. Set COPILOT_PROVIDER=ollama to enable generated prose."}
        </p>
      </div>

      <div className="panel copilot-side">
        <h2>What it can do</h2>
        <ul className="copilot-list">
          <li>Explain why a session was flagged, including z-scores and a counterfactual.</li>
          <li>Pull the entity's recent sessions from Postgres.</li>
          <li>Retrieve similar incidents from the vector index.</li>
          <li>Map the predicted attack to MITRE ATT&CK and a response playbook.</li>
          <li>Summarise PR-AUC, precision@1%, calibration and drift.</li>
        </ul>
        <h2>Guardrails</h2>
        <ul className="copilot-list">
          <li>Read-only tools; no containment is executed.</li>
          <li>Answers must cite <code>[A1]</code>-style evidence.</li>
          <li>Prompt-injection phrases in logs are ignored.</li>
          <li>No LLM key required for the deterministic fallback.</li>
        </ul>
      </div>
    </div>
  );
}
