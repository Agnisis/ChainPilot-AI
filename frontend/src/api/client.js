const API_BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export async function createSession() {
  return request("/api/session", { method: "POST" });
}

export async function getSessionStatus(sessionId) {
  return request(`/api/session/${sessionId}`);
}

export async function uploadDataFile(sessionId, file) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);
  return request("/api/upload/data", { method: "POST", body: form });
}

export async function uploadRagFile(sessionId, file) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);
  return request("/api/upload/rag", { method: "POST", body: form });
}

export async function uploadDemoDataset(sessionId, datasetName) {
  return request("/api/upload/demo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, dataset_name: datasetName }),
  });
}


export async function runAnalysis(sessionId, dateColumn = null, targetColumn = null) {
  return request("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      date_column: dateColumn,
      target_column: targetColumn,
    }),
  });
}

export async function getAnalysis(sessionId) {
  return request(`/api/analysis/${sessionId}`);
}

export async function getRecommendations(sessionId, question = null) {
  return request("/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question }),
  });
}

export async function queryRag(sessionId, query) {
  return request("/api/rag/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, query, top_k: 5 }),
  });
}
