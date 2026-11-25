// In-memory (session-only, cleared on refresh) accumulation of past
// evaluation attempts, keyed by reference file + the ACTUAL measure range
// evaluated -- not the raw text in the measure-range inputs, so leaving
// both blank (whole piece) and later typing the piece's full range
// explicitly land in the same bucket (the backend resolves both to the
// same start_measure/end_measure in its response; see app.py).
// A real database is the eventual home for this; this is the client-side
// stand-in until then.

const attemptHistory = {};

function historyKey(refName, startMeasure, endMeasure) {
  if (!refName || startMeasure == null || endMeasure == null) return null;
  return `${refName}::${startMeasure}::${endMeasure}`;
}

function getAttempts(refName, startMeasure, endMeasure) {
  const key = historyKey(refName, startMeasure, endMeasure);
  return key ? (attemptHistory[key] || []) : [];
}

// feedback is the full structured report from /performances (see
// server/evaluate.py's build_report()) -- kept in full, not just the
// three headline scores, so a past attempt can still be inspected in
// detail (per-note verdicts, regions, extras) via renderFeedback().
// `saved` reflects whether THIS attempt was actually persisted to
// performances/ on the backend (opt-in per submission) -- most attempts
// won't be, so `folder` alone isn't a reliable disk reference.
function recordAttempt(refName, startMeasure, endMeasure, folder, saved, feedback) {
  const key = historyKey(refName, startMeasure, endMeasure);
  if (!key) return [];
  if (!attemptHistory[key]) attemptHistory[key] = [];
  attemptHistory[key].push({ folder, saved, feedback });
  return attemptHistory[key];
}

let progressChart = null;

// Renders (or clears, if entries is empty) a line chart of note/rhythm/
// overall accuracy across attempts for one reference+range bucket, x-axis
// being attempt sequence (1st, 2nd, ... try at THIS bucket), not wall-clock
// time -- what matters here is "did it get better try over try", which a
// timestamp axis would distort whenever there's a gap between sessions.
// Clicking a point calls onPointClick(entry) so the caller can show that
// attempt's full detail instead of only the latest one.
function renderProgressChart(canvas, entries, onPointClick) {
  if (progressChart) {
    progressChart.destroy();
    progressChart = null;
  }
  if (!entries.length) return;

  const labels = entries.map((_, i) => `#${i + 1}`);
  const metric = (key) => entries.map((e) => e.feedback.scores[key]);

  progressChart = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Note accuracy", data: metric("note_accuracy"), borderColor: "#2563eb", tension: 0.15 },
        { label: "Rhythm accuracy", data: metric("rhythm_accuracy"), borderColor: "#16a34a", tension: 0.15 },
        { label: "Overall", data: metric("overall"), borderColor: "#dc2626", borderWidth: 3, tension: 0.15 },
      ],
    },
    options: {
      scales: { y: { min: 0, max: 100, title: { display: true, text: "%" } } },
      onClick: (_event, elements) => {
        if (!elements.length || !onPointClick) return;
        onPointClick(entries[elements[0].index]);
      },
    },
  });
}
