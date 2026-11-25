// Renders the structured report returned by the backend's /performances
// endpoint (see server/evaluate.py's build_report()) as a readable
// summary -- scores, per-note verdicts, disputed regions, extra notes --
// mirroring evaluate.py's own CLI report but as HTML instead of text.
function renderFeedback(container, feedback) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const beat = (b) => `${b.measure}:${Number(b.beat).toFixed(2).replace(/\.?0+$/, "")}`;
  const pct = (v) => (v === null || v === undefined ? "n/a" : `${v.toFixed(1)}%`);

  const s = feedback.scores;
  const zeroedStr = s.zeroed_beats.length ? s.zeroed_beats.map(beat).join(", ") : "none";

  let html = `
    <h3>Scores</h3>
    <table>
      <tr><td>Note accuracy</td><td>${pct(s.note_accuracy)}</td><td>(${s.scored_beats.length} scored beats, ${s.zeroed_beats.length} at zero)</td></tr>
      <tr><td>Rhythm accuracy</td><td>${pct(s.rhythm_accuracy)}</td><td></td></tr>
      <tr><td><strong>Overall (70/30)</strong></td><td><strong>${pct(s.overall)}</strong></td><td></td></tr>
      <tr><td>Zero-value beats</td><td colspan="2">${esc(zeroedStr)}</td></tr>
    </table>
  `;

  if (feedback.regions.length) {
    html += `<h3>Disputed regions</h3><ul>`;
    for (const r of feedback.regions) {
      const beatsStr = r.beats_touched.length ? r.beats_touched.map(beat).join(", ") : "(outside the evaluated range)";
      html += `<li>Region #${r.region_num}: beats [${esc(beatsStr)}] -- ${esc(r.label)}</li>`;
    }
    html += `</ul>`;
  }

  html += `<h3>Per-note detail (${feedback.notes.length} reference notes)</h3>
    <table>
      <tr><th>m:beat</th><th>ref t</th><th>pitch</th><th>verdict</th><th>confidence</th><th>timing</th></tr>`;
  for (const n of feedback.notes) {
    const timingStr = n.timing_tier ? `${n.timing_tier} (${(n.timing_dev_frac * 100).toFixed(0)}%)` : "-";
    html += `<tr><td>${beat(n)}</td><td>${n.ref_onset.toFixed(3)}</td><td>${n.pitch}</td><td>${esc(n.verdict)}</td><td>${esc(n.confidence)}</td><td>${esc(timingStr)}</td></tr>`;
  }
  html += `</table>`;

  if (feedback.extra_notes.length) {
    html += `<h3>Extra notes (${feedback.extra_notes.length})</h3><ul>`;
    for (const e of feedback.extra_notes) {
      html += `<li>[${esc(e.model)}] t=${e.onset.toFixed(3)}s midi=${e.pitch} -- ${esc(e.verdict)} (${esc(e.confidence)})</li>`;
    }
    html += `</ul>`;
  }

  container.innerHTML = html;
}
