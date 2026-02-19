const API_BASE = window.location.origin;

const form = document.getElementById("generate-form");
const urlInput = document.getElementById("youtube-url");
const statusEl = document.getElementById("status");
const loadingEl = document.getElementById("loading");
const titleEl = document.getElementById("video-title");
const tocList = document.getElementById("toc-list");
const topicCards = document.getElementById("topic-cards");
const historyList = document.getElementById("history-list");
const downloadBtn = document.getElementById("download-pdf");
const downloadSecondaryBtn = document.getElementById("download-pdf-secondary");
const copyBtn = document.getElementById("copy-notes");
const themeBtn = document.getElementById("toggle-theme");
const refreshHistoryBtn = document.getElementById("refresh-history");

let currentResult = null;

function normalizeErrorMessage(message) {
  const text = String(message || "").trim();
  if (!text) return "Something went wrong.";

  if (/resource_exhausted|quota|rate limit|429/i.test(text)) {
    return "Gemini quota exceeded for your API key/project. Check Google AI Studio quota or billing, then retry.";
  }

  if (text.length > 280) {
    return `${text.slice(0, 280)}...`;
  }

  return text;
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setLoading(isLoading) {
  loadingEl.classList.toggle("hidden", !isLoading);
}

function setMermaidTheme(isDark) {
  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: false,
      theme: isDark ? "dark" : "default",
      securityLevel: "loose",
    });
  }
}

function getThemePreference() {
  return localStorage.getItem("theme") || "light";
}

function applyTheme(theme) {
  document.body.dataset.theme = theme === "dark" ? "dark" : "light";
  localStorage.setItem("theme", theme);
  setMermaidTheme(theme === "dark");
  themeBtn.textContent = theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode";
}

async function generateNotes(url) {
  setStatus("Generating notes...");
  setLoading(true);
  downloadBtn.disabled = true;
  downloadSecondaryBtn.disabled = true;
  copyBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.detail || "Request failed.");
    }

    const data = await response.json();
    currentResult = data;
    await renderResult(data);
    downloadBtn.disabled = false;
    downloadSecondaryBtn.disabled = false;
    copyBtn.disabled = false;
    setStatus("Done.");
    await loadHistory();
  } catch (error) {
    setStatus(normalizeErrorMessage(error.message), true);
  } finally {
    setLoading(false);
  }
}

async function renderResult(data) {
  titleEl.textContent = data.video_title || "Untitled video";

  const toc = Array.isArray(data.table_of_contents)
    ? data.table_of_contents
    : (data.topics || []).map((topic) => topic.title);

  tocList.innerHTML = "";
  toc.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    tocList.appendChild(li);
  });

  topicCards.innerHTML = "";
  const topics = Array.isArray(data.topics) ? data.topics : [];
  for (let i = 0; i < topics.length; i += 1) {
    const topic = topics[i];
    const card = document.createElement("article");
    card.className = "topic-card";

    const header = document.createElement("div");
    header.className = "topic-header";

    const title = document.createElement("div");
    title.className = "topic-title";
    title.textContent = topic.title || `Topic ${i + 1}`;

    const tag = document.createElement("span");
    tag.className = "topic-tag";
    tag.textContent = `Topic ${i + 1}`;

    header.appendChild(title);
    header.appendChild(tag);

    const explanation = document.createElement("p");
    explanation.textContent = topic.explanation || "No explanation provided.";

    const bullets = document.createElement("ul");
    bullets.className = "bullet-list";
    (topic.bullet_notes || []).forEach((note) => {
      const li = document.createElement("li");
      li.textContent = note;
      bullets.appendChild(li);
    });

    const diagramWrapper = document.createElement("div");
    diagramWrapper.className = "diagram-stack";

    const quizList = document.createElement("div");
    quizList.className = "quiz-list";
    (topic.quiz || []).forEach((quiz, index) => {
      const quizItem = document.createElement("div");
      quizItem.className = "quiz-item";

      const qTitle = document.createElement("h4");
      qTitle.textContent = `${index + 1}. ${quiz.question || "Question"}`;

      const options = document.createElement("ul");
      options.className = "quiz-options";
      (quiz.options || []).forEach((option) => {
        const optionItem = document.createElement("li");
        optionItem.textContent = option;
        options.appendChild(optionItem);
      });

      const correct = document.createElement("div");
      correct.className = "correct-answer";
      correct.textContent = `Answer: ${quiz.correct_answer || "Not provided"}`;

      quizItem.appendChild(qTitle);
      quizItem.appendChild(options);
      quizItem.appendChild(correct);
      quizList.appendChild(quizItem);
    });

    card.appendChild(header);
    card.appendChild(explanation);
    card.appendChild(bullets);
    card.appendChild(diagramWrapper);
    card.appendChild(quizList);
    topicCards.appendChild(card);

    await renderTopicDiagrams(diagramWrapper, topic, i);
  }
}

function getTopicDiagrams(topic) {
  const diagrams = Array.isArray(topic.diagrams) ? topic.diagrams : [];
  const normalized = diagrams
    .filter((item) => item && typeof item === "object")
    .map((item) => ({
      title: (item.title || "Diagram").toString(),
      diagram_type: (item.diagram_type || "mermaid").toString(),
      mermaid: (item.mermaid || "").toString(),
    }))
    .filter((item) => item.mermaid.trim().length > 0);

  if (normalized.length > 0) {
    return normalized;
  }

  const fallback = (topic.diagram || "").toString().trim();
  if (!fallback) {
    return [];
  }

  return [{ title: "Diagram", diagram_type: "mermaid", mermaid: fallback }];
}

async function renderTopicDiagrams(container, topic, index) {
  const diagrams = getTopicDiagrams(topic);

  if (!diagrams.length) {
    const empty = document.createElement("div");
    empty.className = "diagram";
    empty.textContent = "No diagram provided.";
    container.appendChild(empty);
    return;
  }

  for (let i = 0; i < diagrams.length; i += 1) {
    const item = diagrams[i];
    const shell = document.createElement("section");
    shell.className = "diagram-shell";

    const meta = document.createElement("div");
    meta.className = "diagram-meta";
    meta.textContent = `${item.title} (${item.diagram_type})`;

    const body = document.createElement("div");
    body.className = "diagram";

    shell.appendChild(meta);
    shell.appendChild(body);
    container.appendChild(shell);

    await renderDiagram(body, item.mermaid, `${index}-${i}`);
  }
}

async function renderDiagram(container, diagramText, index) {
  const text = (diagramText || "").trim();
  if (!text) {
    container.textContent = "No diagram provided.";
    return;
  }

  const looksLikeMermaid = /flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram|journey|gantt|pie|mindmap|timeline|quadrantChart|requirementDiagram/i.test(text);

  if (!looksLikeMermaid || !window.mermaid) {
    const pre = document.createElement("pre");
    pre.textContent = text;
    container.appendChild(pre);
    return;
  }

  try {
    const { svg } = await mermaid.render(`diagram-${index}-${Date.now()}`, text);
    container.innerHTML = svg;
  } catch (error) {
    const pre = document.createElement("pre");
    pre.textContent = text;
    container.appendChild(pre);
  }
}

async function loadHistory() {
  try {
    const response = await fetch(`${API_BASE}/api/history`);
    if (!response.ok) {
      throw new Error("Failed to load history.");
    }
    const data = await response.json();
    renderHistory(data);
  } catch (error) {
    renderHistory([]);
  }
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.textContent = "No history yet.";
    historyList.appendChild(li);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.className = "history-item";

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.video_title || "Untitled video";

    const meta = document.createElement("small");
    meta.textContent = `${item.created_at} | ${item.video_url}`;

    button.appendChild(meta);
    button.addEventListener("click", () => loadHistoryItem(item.id));

    li.appendChild(button);
    historyList.appendChild(li);
  });
}

async function loadHistoryItem(id) {
  if (!id) return;

  setStatus("Loading saved notes...");
  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/api/history/${id}`);
    if (!response.ok) {
      throw new Error("Failed to load this entry.");
    }
    const data = await response.json();
    currentResult = data;
    await renderResult(data);
    downloadBtn.disabled = false;
    downloadSecondaryBtn.disabled = false;
    copyBtn.disabled = false;
    setStatus("Loaded from history.");
  } catch (error) {
    setStatus(normalizeErrorMessage(error.message), true);
  } finally {
    setLoading(false);
  }
}

function downloadPdf(data) {
  if (!window.jspdf) {
    setStatus("PDF library failed to load.", true);
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const pageHeight = doc.internal.pageSize.getHeight();
  let y = 16;

  const ensureSpace = (needed = 8) => {
    if (y + needed > pageHeight - 12) {
      doc.addPage();
      y = 16;
    }
  };

  const addHeading = (text) => {
    ensureSpace(10);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text(text, 15, y);
    y += 8;
  };

  const addSubheading = (text) => {
    ensureSpace(8);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.text(text, 15, y);
    y += 6;
  };

  const addBody = (text, size = 10) => {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(size);
    const lines = doc.splitTextToSize(text, 180);
    lines.forEach((line) => {
      ensureSpace(6);
      doc.text(line, 15, y);
      y += 5;
    });
    y += 2;
  };

  const addBulletList = (items) => {
    if (!items.length) return;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    items.forEach((item) => {
      const lines = doc.splitTextToSize(`• ${item}`, 175);
      lines.forEach((line) => {
        ensureSpace(6);
        doc.text(line, 18, y);
        y += 5;
      });
    });
    y += 2;
  };

  addHeading(data.video_title || "YouTube Study Notes");
  addBody(`Video URL: ${data.video_url || ""}`, 9);
  addSubheading("Table of Contents");
  addBulletList(data.table_of_contents || []);

  (data.topics || []).forEach((topic, index) => {
    addHeading(`Topic ${index + 1}: ${topic.title || "Untitled"}`);
    addSubheading("Explanation");
    addBody(topic.explanation || "No explanation provided.");
    addSubheading("Key Notes");
    addBulletList(topic.bullet_notes || []);
    addSubheading("Diagrams");
    const diagrams = getTopicDiagrams(topic);
    if (diagrams.length) {
      diagrams.forEach((diagram, diagramIndex) => {
        addBody(`${diagramIndex + 1}. ${diagram.title} (${diagram.diagram_type})`, 10);
        addBody(diagram.mermaid || "", 9);
      });
    } else {
      addBody("No diagram provided.");
    }

    if ((topic.quiz || []).length) {
      addSubheading("Quiz");
      (topic.quiz || []).forEach((quiz, qIndex) => {
        addBody(`Q${qIndex + 1}. ${quiz.question || ""}`, 10);
        addBulletList(quiz.options || []);
        addBody(`Answer: ${quiz.correct_answer || ""}`, 9);
      });
    }
  });

  const safeTitle = (data.video_title || "notes")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
  doc.save(`${safeTitle || "notes"}.pdf`);
}

function formatNotesText(data) {
  const lines = [];
  lines.push(`Video Title: ${data.video_title || "Untitled"}`);
  lines.push(`Video URL: ${data.video_url || ""}`);
  lines.push("");
  lines.push("Table of Contents:");
  (data.table_of_contents || []).forEach((item, index) => {
    lines.push(`${index + 1}. ${item}`);
  });

  (data.topics || []).forEach((topic, index) => {
    lines.push("");
    lines.push(`Topic ${index + 1}: ${topic.title || "Untitled"}`);
    lines.push(`Explanation: ${topic.explanation || ""}`);
    lines.push("Notes:");
    (topic.bullet_notes || []).forEach((note) => {
      lines.push(`- ${note}`);
    });
    lines.push("Diagrams:");
    const diagrams = getTopicDiagrams(topic);
    if (diagrams.length) {
      diagrams.forEach((diagram, diagramIndex) => {
        lines.push(`${diagramIndex + 1}. ${diagram.title} (${diagram.diagram_type})`);
        lines.push(diagram.mermaid || "");
      });
    } else {
      lines.push("No diagram provided.");
    }
    lines.push("Quiz:");
    (topic.quiz || []).forEach((quiz, qIndex) => {
      lines.push(`Q${qIndex + 1}. ${quiz.question || ""}`);
      (quiz.options || []).forEach((option) => {
        lines.push(`  * ${option}`);
      });
      lines.push(`Answer: ${quiz.correct_answer || ""}`);
    });
  });

  return lines.join("\n");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const url = urlInput.value.trim();
  if (!url) {
    setStatus("Please paste a YouTube URL.", true);
    return;
  }
  generateNotes(url);
});

downloadBtn.addEventListener("click", () => {
  if (currentResult) {
    downloadPdf(currentResult);
  }
});

downloadSecondaryBtn.addEventListener("click", () => {
  if (currentResult) {
    downloadPdf(currentResult);
  }
});

copyBtn.addEventListener("click", async () => {
  if (!currentResult) return;
  const text = formatNotesText(currentResult);

  try {
    await navigator.clipboard.writeText(text);
    setStatus("Notes copied to clipboard.");
  } catch (error) {
    setStatus("Clipboard blocked. Use HTTPS or try another browser.", true);
  }
});

themeBtn.addEventListener("click", () => {
  const currentTheme = getThemePreference();
  const nextTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
  if (currentResult) {
    renderResult(currentResult);
  }
});

refreshHistoryBtn.addEventListener("click", () => {
  loadHistory();
});

applyTheme(getThemePreference());
loadHistory();
