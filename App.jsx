import React, { useEffect, useState } from "react";

const API = "http://localhost:8000";

const SAMPLES = [
  {
    label: "Stabbing after planned attack",
    text: `On 12 March 2023, Suresh, who had a long-standing land dispute with his neighbour Mohan, bought a knife two days earlier and waited outside Mohan's house at night. When Mohan came out, Suresh stabbed him four times in the chest saying "today I will finish you". Mohan died on the way to hospital. Two neighbours who witnessed the attack identified Suresh in court. The post-mortem report confirmed death due to stab wounds to the heart, and the knife with Mohan's blood was recovered from Suresh's house at his instance.`,
  },
  {
    label: "Sudden fight, single blow",
    text: `During a sudden quarrel over a cricket match, Arjun and Vikas began abusing each other. Vikas slapped Arjun, who in the heat of the moment picked up a wooden stick lying nearby and hit Vikas once on the head. Vikas fell, was taken to hospital and died two days later of a head injury. There was no previous enmity between them. Three spectators gave consistent statements, and the medical report confirmed that the head injury caused death.`,
  },
  {
    label: "Phone snatching, weak evidence",
    text: `Priya complained that a man on a motorcycle snatched her mobile phone near the bus stand in the evening and fled. Ramesh was arrested a week later and the police claimed to recover a phone from him. However, the phone's IMEI number in the recovery memo did not match the purchase bill, the only independent witness turned hostile in court, and Priya could not identify Ramesh with certainty in the test identification parade.`,
  },
  {
    label: "Fake job offer fraud",
    text: `Anil told several young men that he could get them railway jobs through his contacts and showed them forged appointment letters bearing a fake railway seal. Believing him, Kiran and four others paid him Rs. 3 lakh each by bank transfer. No jobs were given and Anil stopped answering calls. Bank statements show the transfers into Anil's account, the letters were confirmed as forged by the railway department, and the victims identified Anil in court.`,
  },
  {
    label: "Dowry death",
    text: `Meena married Rakesh in 2021. Her parents testified that Rakesh and his mother repeatedly demanded a car and Rs. 5 lakh as dowry and beat Meena when the demand was not met, and Meena had told them about the harassment over the phone a week before her death. In August 2022, fourteen months after the marriage, Meena died of burn injuries at her matrimonial home. The post-mortem showed kerosene burns, and neighbours had heard quarrels about money on the night of the incident.`,
  },
  {
    label: "Self-defence claim",
    text: `Raju was attacked at night by Shankar, who had a history of violent assaults and came at him with a sickle, injuring Raju's arm. While struggling, Raju snatched an iron rod and struck Shankar once on the head; Shankar died. Raju immediately went to the police station and reported the incident. The medical report of Raju shows a deep sickle wound on his forearm, and an eyewitness confirmed that Shankar attacked first.`,
  },
];

const VERDICT_STYLE = {
  CONVICTED: { cls: "convicted", icon: "⚖", sub: "The accused is likely to be found guilty" },
  ACQUITTED: { cls: "acquitted", icon: "✓", sub: "The accused is likely to be acquitted" },
  INCONCLUSIVE: { cls: "inconclusive", icon: "?", sub: "Not enough information to decide" },
};

const STATUS_ICON = { proved: "✓", not_proved: "✗", unclear: "?" };

function pct(x) {
  return x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`;
}

function yesNo(v, yes, no) {
  if (v === true) return yes;
  if (v === false) return no;
  return "Verify";
}

function PunishmentCard({ p, applies }) {
  return (
    <div className="punishment">
      <div className="punishment-head">
        <span className="sec">IPC {p.section}</span>
        <span className="sec-title">{p.title}</span>
      </div>
      {p.defined_in && (
        <div className="muted small">Offence defined in s.{p.defined_in}, punishable under s.{p.section}.</div>
      )}
      {p.punishable_under && (
        <div className="muted small">
          Punishable under s.{p.punishable_under}
          {p.punishable_under_title ? ` (${p.punishable_under_title})` : ""}.
        </div>
      )}
      <div className={`punishment-summary ${applies ? "" : "dim"}`}>{p.summary}</div>
      <div className="punishment-text">{p.punishment_text}</div>
      <div className="chips">
        {p.death && <span className="chip danger">Death penalty possible</span>}
        {p.life && <span className="chip warn">Life imprisonment possible</span>}
        {p.min_years && <span className="chip warn">Mandatory minimum</span>}
        {p.imprisonment_kind === "rigorous" && <span className="chip">Rigorous imprisonment</span>}
        {p.imprisonment_kind === "simple" && <span className="chip">Simple imprisonment</span>}
        <span className="chip">{yesNo(p.cognizable, "Cognizable", "Non-cognizable")}</span>
        <span className="chip">{yesNo(p.bailable, "Bailable", "Non-bailable")}</span>
        {p.triable_by && <span className="chip">Triable by {p.triable_by}</span>}
        <span className="chip">Severity {p.severity}/10</span>
      </div>
      {p.note && <div className="note">Note: {p.note}</div>}
    </div>
  );
}

function ElementList({ charge }) {
  if (!charge) return null;
  return (
    <div className="card">
      <h3>
        Legal ingredients — IPC {charge.section} {charge.title}
      </h3>
      <ul className="elements">
        {charge.elements.map((e) => (
          <li key={e.id} className={e.status}>
            <span className="status">{STATUS_ICON[e.status]}</span>
            <div>
              <div>{e.text}</div>
              {e.evidence && <div className="evidence-quote">“{e.evidence}”</div>}
            </div>
          </li>
        ))}
      </ul>
      {charge.note && <p className="muted small">{charge.note}</p>}
    </div>
  );
}

function Evidence({ ev, exception }) {
  return (
    <div className="card">
      <h3>Evidence assessment</h3>
      <div className="meter">
        <div className="meter-label">
          <span>Evidence strength</span>
          <span>{pct(ev.score)}</span>
        </div>
        <div className="bar">
          <div className="fill" style={{ width: pct(ev.score) }} />
        </div>
      </div>
      <div className="two-col">
        <div>
          <h4>Supports prosecution</h4>
          {ev.prosecution.length === 0 && <p className="muted small">None described</p>}
          {ev.prosecution.map((e, i) => (
            <div key={i} className="ev pro">
              <b>{e.type.replace(/_/g, " ")}</b> <span className="muted">({e.strength})</span>
              {e.detail && <div className="small">{e.detail}</div>}
            </div>
          ))}
        </div>
        <div>
          <h4>Weakens prosecution</h4>
          {ev.defence.length === 0 && <p className="muted small">None described</p>}
          {ev.defence.map((e, i) => (
            <div key={i} className="ev con">
              <b>{e.type.replace(/_/g, " ")}</b> <span className="muted">({e.strength})</span>
              {e.detail && <div className="small">{e.detail}</div>}
            </div>
          ))}
        </div>
      </div>
      {ev.note && <p className="muted small">{ev.note}</p>}
      {exception && exception.type !== "none" && (
        <div className={`exception ${exception.established}`}>
          <b>General exception claimed:</b> {exception.law || exception.type} —{" "}
          {exception.established === "yes" ? "established" : exception.established === "no" ? "not established" : "unclear"}
          {exception.detail && <div className="small">{exception.detail}</div>}
        </div>
      )}
    </div>
  );
}

function Result({ r }) {
  const v = VERDICT_STYLE[r.verdict] || VERDICT_STYLE.INCONCLUSIVE;
  const mainCharge = r.charges[0] || r.alleged_charge;
  return (
    <>
      <div className={`verdict ${v.cls}`}>
        <div className="verdict-icon">{v.icon}</div>
        <div className="verdict-body">
          <div className="verdict-title">{r.verdict}</div>
          <div className="verdict-sub">{v.sub}</div>
          <div className="verdict-reason">{r.verdict_reason}</div>
        </div>
        {r.conviction_probability !== null && (
          <div className="verdict-stats">
            <div className="big">{pct(r.conviction_probability)}</div>
            <div className="muted small">conviction probability</div>
            <div className="small">Confidence: {r.confidence_level}</div>
          </div>
        )}
      </div>

      {r.reduced_from && (
        <div className="notice">
          Charge reduced from IPC {r.reduced_from.section} ({r.reduced_from.title}) because these ingredients were not proved:{" "}
          {r.reduced_from.failed_elements.join("; ")}.
        </div>
      )}

      {r.punishment.length > 0 && (
        <div className="card">
          <h3>
            {r.punishment_applies
              ? r.punishment.length > 1
                ? "Charges and punishment"
                : "Charge and punishment"
              : "Punishment prescribed for the alleged offence (not imposed)"}
          </h3>
          {r.punishment.map((p) => (
            <PunishmentCard key={p.section} p={p} applies={r.punishment_applies} />
          ))}
          {r.punishment_applies && (
            <p className="muted small">
              The court decides the actual sentence within this range, considering aggravating and mitigating
              circumstances.
            </p>
          )}
        </div>
      )}

      <div className="card">
        <h3>Reasoning</h3>
        {r.summary && (
          <p>
            <b>Case summary:</b> {r.summary}
          </p>
        )}
        <p className="rationale">{r.rationale}</p>
        <div className="scores">
          <span>Symbolic: {pct(r.symbolic_probability)}</span>
          <span>Neural (LLM): {pct(r.neural_probability)}</span>
          <span>Fused: {pct(r.conviction_probability)}</span>
          <span className="badge">{r.mode.replace(/_/g, " ")}</span>
        </div>
      </div>

      <ElementList charge={mainCharge} />
      <Evidence ev={r.evidence} exception={r.general_exception} />

      <details>
        <summary>Candidate sections considered ({r.candidates.length})</summary>
        <table>
          <thead>
            <tr>
              <th>Section</th>
              <th>Title</th>
              <th>Retrieval</th>
              <th>Ingredients</th>
              <th>LLM fit</th>
              <th>Score</th>
              <th>Applies</th>
            </tr>
          </thead>
          <tbody>
            {r.candidates.map((c) => (
              <tr key={c.section}>
                <td>{c.section}</td>
                <td>{c.title}</td>
                <td>{c.retrieval_score.toFixed(2)}</td>
                <td>
                  {c.proved}✓ {c.unclear}? {c.not_proved}✗
                </td>
                <td>{c.llm_fit}</td>
                <td>{c.fused_score.toFixed(2)}</td>
                <td>{c.applicable ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <details>
        <summary>Reasoning trace</summary>
        <ol className="trace">
          {r.reasoning_trace.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ol>
      </details>

      {r.facts && (
        <details>
          <summary>Extracted facts</summary>
          <pre>{JSON.stringify(r.facts, null, 2)}</pre>
        </details>
      )}

      <p className="disclaimer">{r.disclaimer}</p>
    </>
  );
}

export default function App() {
  const [caseText, setCaseText] = useState(SAMPLES[0].text);
  const [useLlm, setUseLlm] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_text: caseText, use_llm: useLlm }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);
      setResult(data);
    } catch (err) {
      setError(err.message === "Failed to fetch" ? "Cannot reach the backend at " + API : err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Neuro-Symbolic Legal Judgment Prediction</h1>
      <p className="subtitle">
        Criminal case analysis over {health ? health.sections : 213} IPC sections — LLM fact reading + rule-based
        legal reasoning
      </p>

      {health && !health.llm_available && (
        <div className="notice">LLM unavailable — running in rule-based (symbolic-only) mode.</div>
      )}

      <div className="toolbar">
        <select
          onChange={(e) => {
            if (e.target.value !== "") setCaseText(SAMPLES[+e.target.value].text);
          }}
          defaultValue="0"
        >
          {SAMPLES.map((s, i) => (
            <option key={i} value={i}>
              Sample: {s.label}
            </option>
          ))}
        </select>
        <label className="toggle">
          <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
          Use LLM
        </label>
      </div>

      <textarea
        value={caseText}
        onChange={(e) => setCaseText(e.target.value)}
        rows={9}
        placeholder="Describe the case: what the accused did, their intention, the result, and the evidence (witnesses, medical reports, recoveries, contradictions)..."
      />
      <div className="muted small">
        For a better assessment, include the act, the accused's intention, the consequences, and what evidence exists.
      </div>

      <button onClick={handlePredict} disabled={loading || caseText.trim().length < 30}>
        {loading ? "Analyzing… (up to ~30s)" : "Assess case"}
      </button>

      {error && <div className="error">{error}</div>}
      {result && <Result r={result} />}
    </div>
  );
}
