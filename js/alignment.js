// Dynamic Time Warping alignment between the performance's detected onset
// sequence (with chroma vectors) and the reference template's expected
// onset sequence (with chroma vectors). This is the "verify against a known
// template" problem, not open-ended transcription.

function cosineDistance(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 1;
  return 1 - dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Penalty added to a non-diagonal DTW step (one performance onset absorbing
// two reference notes, or vice versa). Without this, the cost function was
// pure chroma distance with zero disincentive for skipping around -- so any
// near-tie in chroma similarity let the path drift off a well-timed 1:1
// match in exchange for a negligibly better chroma score, producing large,
// suspiciously constant timing-deviation offsets that were actually a
// single mis-stepped alignment index, not real tempo instability (confirmed
// by inspecting a real take where a well-timed, high-amplitude onset was
// skipped in favor of a worse-timed one with marginally closer chroma).
const DTW_SKIP_PENALTY = 0.4;

/**
 * Standard DTW over two sequences of chroma vectors. Returns the warping
 * path as pairs of indices [perfIndex, refIndex] and the total cost.
 */
function dtwAlign(perfChroma, refChroma) {
  const n = perfChroma.length;
  const m = refChroma.length;
  if (n === 0 || m === 0) return { path: [], cost: Infinity };

  const cost = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(Infinity));
  cost[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const d = cosineDistance(perfChroma[i - 1], refChroma[j - 1]);
      cost[i][j] = d + Math.min(
        cost[i - 1][j - 1],
        cost[i - 1][j] + DTW_SKIP_PENALTY,
        cost[i][j - 1] + DTW_SKIP_PENALTY
      );
    }
  }

  // Backtrack from (n, m) to (0, 0).
  const path = [];
  let i = n, j = m;
  while (i > 0 && j > 0) {
    path.push([i - 1, j - 1]);
    const diag = cost[i - 1][j - 1];
    const up = cost[i - 1][j] + DTW_SKIP_PENALTY;
    const left = cost[i][j - 1] + DTW_SKIP_PENALTY;
    const minVal = Math.min(diag, up, left);
    if (minVal === diag) { i--; j--; }
    else if (minVal === up) { i--; }
    else { j--; }
  }
  path.reverse();

  return { path, cost: cost[n][m] };
}
