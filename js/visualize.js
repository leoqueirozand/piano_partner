// Renders the expected-vs-detected onset timeline, the pitch contour, and
// the raw diff JSON onto the page.

function drawOnsetTimeline(canvas, reference, performance, diff) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Blue dots and connecting lines must share one time origin: the diff's
  // anchor-and-tempo-adjusted expectedOnsetSec, not raw reference.onsets_sec
  // (which is on a different, un-anchored clock and would draw dots that
  // don't line up with where the lines actually start).
  const perNoteByRef = [...diff.perNote].sort((a, b) => a.refIdx - b.refIdx);

  const maxTime = Math.max(
    Math.max(0, ...perNoteByRef.map((n) => n.expectedOnsetSec)),
    performance.onsets[performance.onsets.length - 1] || 1
  ) + 1;
  const xScale = (t) => (t / maxTime) * (w - 20) + 10;

  // Reference onsets (top row, blue) vs performance onsets (bottom row, orange).
  const refY = h * 0.35, perfY = h * 0.75;

  ctx.strokeStyle = "#ccc";
  ctx.beginPath();
  ctx.moveTo(0, refY); ctx.lineTo(w, refY);
  ctx.moveTo(0, perfY); ctx.lineTo(w, perfY);
  ctx.stroke();

  ctx.fillStyle = "#2563eb";
  perNoteByRef.forEach((n) => {
    ctx.beginPath();
    ctx.arc(xScale(n.expectedOnsetSec), refY, 4, 0, 2 * Math.PI);
    ctx.fill();
  });

  ctx.fillStyle = "#ea580c";
  performance.onsets.forEach((t) => {
    ctx.beginPath();
    ctx.arc(xScale(t), perfY, 4, 0, 2 * Math.PI);
    ctx.fill();
  });

  // Draw matched pairs as thin connecting lines, colored by deviation size.
  diff.perNote.forEach((n) => {
    const dev = Math.abs(n.timingDeviationSec);
    ctx.strokeStyle = dev > 0.15 ? "#dc2626" : dev > 0.05 ? "#f59e0b" : "#16a34a";
    ctx.beginPath();
    ctx.moveTo(xScale(n.expectedOnsetSec), refY);
    ctx.lineTo(xScale(n.actualOnsetSec), perfY);
    ctx.stroke();
  });

  ctx.fillStyle = "#333";
  ctx.font = "12px sans-serif";
  ctx.fillText("reference (expected)", 10, refY - 10);
  ctx.fillText("performance (detected)", 10, perfY + 18);
}

function drawPitchContour(canvas, performance) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const points = performance.pitchContour.filter((p) => p.hz > 0);
  if (points.length === 0) {
    ctx.fillText("no pitched frames detected", 10, h / 2);
    return;
  }
  const maxTime = points[points.length - 1].time || 1;
  const minHz = Math.min(...points.map((p) => p.hz));
  const maxHz = Math.max(...points.map((p) => p.hz));
  const xScale = (t) => (t / maxTime) * (w - 20) + 10;
  const yScale = (hz) => h - 10 - ((hz - minHz) / Math.max(maxHz - minHz, 1)) * (h - 20);

  ctx.strokeStyle = "#7c3aed";
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = xScale(p.time), y = yScale(p.hz);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#333";
  ctx.font = "12px sans-serif";
  ctx.fillText("predominant pitch contour (Hz, medium confidence)", 10, 14);
}

function renderDiffJson(preElement, diff) {
  preElement.textContent = JSON.stringify(diff, null, 2);
}

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
      const corrob = r.corroborated_by_basic_pitch ? "corroborated" : "not corroborated";
      html += `<li>Region #${r.region_num}: beats [${esc(beatsStr)}] -- ${esc(r.label)} (${esc(corrob)} by Basic Pitch: ${esc(r.corroboration_detail)})</li>`;
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
