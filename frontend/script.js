function resolveApiBase() {
  const queryApi = new URLSearchParams(window.location.search).get("api");
  if (queryApi) return queryApi.replace(/\/+$/, "");
  if (window.location.protocol === "file:") return "http://127.0.0.1:8000";
  return window.location.origin;
}

const API_BASE = resolveApiBase();

const LANGUAGE_LABELS = {
  en: "English",
  hi: "Hindi",
  es: "Spanish",
  fr: "French",
  de: "German",
  pt: "Portuguese",
  ta: "Tamil",
  te: "Telugu",
  bn: "Bengali",
  ja: "Japanese",
};

const els = {
  authSection: document.getElementById("auth-section"),
  authStatus: document.getElementById("auth-status"),
  authTabLogin: document.getElementById("auth-tab-login"),
  authTabRegister: document.getElementById("auth-tab-register"),
  loginForm: document.getElementById("login-form"),
  registerForm: document.getElementById("register-form"),
  loginBtn: document.getElementById("login-btn"),
  registerBtn: document.getElementById("register-btn"),
  logoutBtn: document.getElementById("logout-btn"),
  sessionPill: document.getElementById("session-pill"),

  studentSection: document.getElementById("student-section"),
  studentList: document.getElementById("student-list"),
  studentName: document.getElementById("student-name"),
  studentMobile: document.getElementById("student-mobile"),
  studentCreated: document.getElementById("student-created"),
  studentAvatar: document.getElementById("student-avatar"),
  studentProfileName: document.getElementById("student-profile-name"),
  studentLastActive: document.getElementById("student-last-active"),
  studentProgressLabel: document.getElementById("student-progress-label"),
  studentProgressBar: document.getElementById("student-progress-bar"),

  secureApp: document.getElementById("secure-app"),
  heroTypingTarget: document.getElementById("hero-typing-target"),
  heroGenerateBtn: document.getElementById("hero-generate-cta"),
  heroDemoBtn: document.getElementById("hero-demo-btn"),
  generateForm: document.getElementById("generate-form"),
  youtubeUrl: document.getElementById("youtube-url"),
  generateBtn: document.getElementById("generate-btn"),
  status: document.getElementById("status"),
  loading: document.getElementById("loading"),
  loadingStep: document.getElementById("loading-step"),
  downloadPdfBtn: document.getElementById("download-pdf-btn"),
  notesEmptyState: document.getElementById("notes-empty-state"),
  emptyGenerateBtn: document.getElementById("empty-generate-btn"),
  notesDataBlocks: Array.from(document.querySelectorAll(".notes-data-block")),

  languageSelect: document.getElementById("language-select"),
  languageStatus: document.getElementById("language-status"),
  videoTitle: document.getElementById("video-title"),
  videoLink: document.getElementById("video-link"),
  tocList: document.getElementById("toc-list"),
  topicCards: document.getElementById("topic-cards"),

  historyList: document.getElementById("history-list"),
  refreshHistoryBtn: document.getElementById("refresh-history"),
  historySearch: document.getElementById("history-search"),
  historyFilter: document.getElementById("history-filter"),
  historyTags: document.getElementById("history-tags"),
  historyCount: document.getElementById("history-count"),
  statTotalNotes: document.getElementById("stat-total-notes"),
  statTimeSaved: document.getElementById("stat-time-saved"),
  statSubjectsCovered: document.getElementById("stat-subjects-covered"),
  feedbackForm: document.getElementById("feedback-form"),
  feedbackName: document.getElementById("feedback-name"),
  feedbackEmail: document.getElementById("feedback-email"),
  feedbackMessage: document.getElementById("feedback-message"),
  feedbackStatus: document.getElementById("feedback-status"),
  feedbackSubmitBtn: document.getElementById("feedback-submit-btn"),
  demoModal: document.getElementById("demo-modal"),
  closeDemoBtn: document.getElementById("close-demo-btn"),
  themeToggle: document.getElementById("theme-toggle"),
};

const state = {
  authMode: "login",
  authToken: localStorage.getItem("studykit_auth_token") || "",
  currentStudent: null,
  sourceResult: null,
  currentResult: null,
  activeLanguage: "en",
  translationCache: {},
  loadingSequenceTimer: null,
  historyItems: [],
  hiddenHistoryIds: new Set(JSON.parse(localStorage.getItem("studykit_hidden_history_ids") || "[]")),
  historySearch: "",
  historyFilter: "all",
  historyTag: "all",
  heroTypingPhrases: ["Exam-Ready Notes", "Revision-Friendly Notes", "AI Study Packs"],
  heroTypingTimer: null,
  revealObserver: null,
};

function setStatus(message, isError = false) {
  if (!els.status) return;
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

function setAuthStatus(message, isError = false) {
  if (!els.authStatus) return;
  els.authStatus.textContent = message;
  els.authStatus.classList.toggle("error", isError);
}

function setLanguageStatus(message, isError = false) {
  if (!els.languageStatus) return;
  els.languageStatus.textContent = message;
  els.languageStatus.classList.toggle("error", isError);
}

function updateLanguageStatus() {
  const name = LANGUAGE_LABELS[state.activeLanguage] || "English";
  setLanguageStatus(`Showing notes in ${name}.`);
}

function setLoading(isLoading) {
  if (!els.loading) return;
  els.loading.classList.toggle("hidden", !isLoading);

  if (!isLoading) {
    stopLoadingSequence();
  }
}

function startLoadingSequence(messages = []) {
  stopLoadingSequence();

  const safeMessages = Array.isArray(messages) && messages.length
    ? messages
    : ["Analyzing lecture...", "Extracting key points...", "Building notes..."];

  let index = 0;

  if (els.loadingStep) {
    els.loadingStep.textContent = safeMessages[0];
  }

  state.loadingSequenceTimer = window.setInterval(() => {
    index = (index + 1) % safeMessages.length;
    if (els.loadingStep) {
      els.loadingStep.textContent = safeMessages[index];
    }
  }, 1350);
}

function stopLoadingSequence() {
  if (state.loadingSequenceTimer) {
    window.clearInterval(state.loadingSequenceTimer);
    state.loadingSequenceTimer = null;
  }

  if (els.loadingStep) {
    els.loadingStep.textContent = "Preparing your study notes...";
  }
}

function setGenerateBusy(isBusy) {
  if (els.generateBtn) els.generateBtn.disabled = isBusy;
  if (els.youtubeUrl) els.youtubeUrl.disabled = isBusy;
  setButtonLoading(els.generateBtn, isBusy, "Generating...");
  if (els.downloadPdfBtn && !state.currentResult) els.downloadPdfBtn.disabled = true;
}

function setAuthBusy(isBusy, mode) {
  if (mode === "login" && els.loginBtn) {
    els.loginBtn.disabled = isBusy;
    setButtonLoading(els.loginBtn, isBusy, "Signing In...");
  }

  if (mode === "register" && els.registerBtn) {
    els.registerBtn.disabled = isBusy;
    setButtonLoading(els.registerBtn, isBusy, "Creating Account...");
  }
}

function setExportEnabled(enabled) {
  if (!els.downloadPdfBtn) return;
  els.downloadPdfBtn.disabled = !enabled;
}

function setLanguageEnabled(enabled) {
  if (els.languageSelect) els.languageSelect.disabled = !enabled;
}

function setButtonLoading(button, isLoading, loadingLabel) {
  if (!button) return;
  if (!button.dataset.defaultLabel) {
    button.dataset.defaultLabel = button.textContent || "";
  }

  button.dataset.loading = isLoading ? "true" : "false";
  button.setAttribute("aria-busy", isLoading ? "true" : "false");

  if (isLoading && loadingLabel) {
    button.textContent = loadingLabel;
    return;
  }

  button.textContent = button.dataset.defaultLabel || "";
}

function setNotesContentVisibility(hasContent) {
  if (Array.isArray(els.notesDataBlocks)) {
    els.notesDataBlocks.forEach((block) => {
      block.classList.toggle("hidden", !hasContent);
    });
  }

  if (els.notesEmptyState) {
    els.notesEmptyState.classList.toggle("hidden", hasContent);
  }
}

function focusGenerateWorkspace() {
  els.generateForm?.scrollIntoView({ behavior: "smooth", block: "start" });
  window.setTimeout(() => {
    els.youtubeUrl?.focus();
  }, 220);
}

function renderNotesEmptyState() {
  if (els.videoTitle) {
    els.videoTitle.textContent = "Paste a lecture link to generate smart notes.";
  }

  if (els.videoLink) {
    els.videoLink.href = "#";
    els.videoLink.textContent = "Use the generator to start your next study session.";
    els.videoLink.classList.add("disabled-link");
  }

  if (els.tocList) els.tocList.innerHTML = "";
  if (els.topicCards) els.topicCards.innerHTML = "";
  setNotesContentVisibility(false);
}

function showNotesSkeleton(title = "Generating smart notes...") {
  if (els.videoTitle) {
    els.videoTitle.textContent = title;
  }

  if (els.videoLink) {
    els.videoLink.href = "#";
    els.videoLink.textContent = "Please wait while we prepare the lecture structure.";
    els.videoLink.classList.add("disabled-link");
  }

  setNotesContentVisibility(true);

  if (els.tocList) {
    els.tocList.innerHTML = Array.from({ length: 4 }, () => `
      <li class="skeleton-card">
        <span class="skeleton-line medium"></span>
        <span class="skeleton-line short"></span>
      </li>
    `).join("");
  }

  if (els.topicCards) {
    els.topicCards.innerHTML = Array.from({ length: 3 }, () => `
      <article class="skeleton-card">
        <span class="skeleton-line short"></span>
        <span class="skeleton-line tall"></span>
        <span class="skeleton-line"></span>
        <span class="skeleton-line medium"></span>
        <span class="skeleton-line"></span>
      </article>
    `).join("");
  }
}

function setFeedbackStatus(message, kind = "info") {
  if (!els.feedbackStatus) return;
  els.feedbackStatus.textContent = message;
  els.feedbackStatus.classList.remove("error", "success");
  if (kind === "error" || kind === "success") {
    els.feedbackStatus.classList.add(kind);
  }
}

function normalizeErrorMessage(message) {
  const text = String(message || "").trim();
  if (!text) return "Something went wrong.";

  const lower = text.toLowerCase();
  if (
    lower === "failed to fetch" ||
    lower === "networkerror when attempting to fetch resource." ||
    lower === "network request failed" ||
    lower === "load failed"
  ) {
    return `Unable to connect to backend (${API_BASE}). Start uvicorn and open this page from the same host and port.`;
  }

  if (/resource_exhausted|quota|rate limit|429/i.test(text)) {
    return "Gemini quota exceeded for your API key/project. Check Google AI Studio quota or billing, then retry.";
  }

  if (text.length > 280) return `${text.slice(0, 280)}...`;
  return text;
}

function formatIsoDate(value) {
  const raw = String(value || "").trim();
  if (!raw) return "-";
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return raw;
  return parsed.toLocaleString();
}

function inferHistorySubject(item) {
  const text = `${item?.video_title || ""} ${item?.video_url || ""}`.toLowerCase();
  const subjectRules = [
    ["Physics", /physics|quantum|gauss|electric|electro|thermo|mechanics|wave|motion/],
    ["Chemistry", /chemistry|organic|inorganic|molecule|atomic|bond|reaction/],
    ["Mathematics", /math|mathematics|algebra|calculus|geometry|trigonometry|probability/],
    ["Biology", /biology|cell|genetics|photosynthesis|human body|anatomy|botany|zoology/],
    ["Computer Science", /computer|coding|programming|python|java|algorithm|data structure|ai|machine learning/],
    ["History", /history|civilization|war|empire|ancient|medieval|modern history/],
    ["Economics", /economics|economy|macro|micro|finance|market|trade/],
    ["English", /english|grammar|literature|poetry|writing|communication/],
  ];

  const match = subjectRules.find(([, pattern]) => pattern.test(text));
  return match ? match[0] : "General";
}

function getHistoryDateBucket(value) {
  const raw = String(value || "").trim();
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return "older";

  const now = new Date();
  const diffMs = now.getTime() - parsed.getTime();
  const diffDays = diffMs / (1000 * 60 * 60 * 24);

  if (diffDays < 1) return "today";
  if (diffDays < 7) return "week";
  if (diffDays < 31) return "month";
  return "older";
}

function extractYouTubeVideoId(url) {
  const raw = String(url || "").trim();
  if (!raw) return "";

  try {
    const parsed = new URL(raw);
    const host = parsed.hostname.replace(/^www\./, "").toLowerCase();

    if (host === "youtu.be") {
      return parsed.pathname.replace(/\//g, "").slice(0, 11);
    }

    if (host.includes("youtube.com")) {
      const watchId = parsed.searchParams.get("v");
      if (watchId) return watchId.slice(0, 11);

      const pathParts = parsed.pathname.split("/").filter(Boolean);
      const embedIndex = pathParts.findIndex((part) => ["embed", "shorts", "live"].includes(part));
      if (embedIndex >= 0 && pathParts[embedIndex + 1]) {
        return pathParts[embedIndex + 1].slice(0, 11);
      }
    }
  } catch (error) {
    return "";
  }

  return "";
}

function enrichHistoryItem(item) {
  const subject = inferHistorySubject(item);
  const createdLabel = formatIsoDate(item?.created_at);
  const title = String(item?.video_title || "Untitled video").trim() || "Untitled video";
  const videoUrl = String(item?.video_url || "").trim();
  const videoId = extractYouTubeVideoId(videoUrl);
  const thumbnailUrl = videoId ? `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg` : "";

  return {
    ...item,
    subject,
    createdLabel,
    dateBucket: getHistoryDateBucket(item?.created_at),
    title,
    videoUrl,
    videoId,
    thumbnailUrl,
    searchText: `${title} ${videoUrl} ${subject}`.toLowerCase(),
  };
}

function formatEstimatedTimeSaved(totalItems) {
  const minutes = Math.max(0, Number(totalItems || 0)) * 35;
  if (minutes >= 120) return `${Math.round(minutes / 60)} hrs`;
  if (minutes >= 60) return `${(minutes / 60).toFixed(1)} hrs`;
  return `${minutes} min`;
}

function updateDashboardStats(items) {
  const safeItems = Array.isArray(items) ? items : [];
  const subjects = new Set(safeItems.map((item) => item.subject).filter(Boolean));

  if (els.statTotalNotes) els.statTotalNotes.textContent = String(safeItems.length);
  if (els.statTimeSaved) els.statTimeSaved.textContent = formatEstimatedTimeSaved(safeItems.length);
  if (els.statSubjectsCovered) els.statSubjectsCovered.textContent = String(subjects.size);
}

function renderHistoryTags(items) {
  if (!els.historyTags) return;

  const safeItems = Array.isArray(items) ? items : [];
  const subjects = Array.from(new Set(safeItems.map((item) => item.subject).filter(Boolean))).sort();
  const values = ["all", ...subjects];

  if (!values.includes(state.historyTag)) {
    state.historyTag = "all";
  }

  els.historyTags.innerHTML = "";

  values.forEach((value) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-tag-btn";
    button.textContent = value === "all" ? "All Subjects" : value;
    button.classList.toggle("active", state.historyTag === value);
    button.addEventListener("click", () => {
      state.historyTag = value;
      renderHistoryTags(state.historyItems);
      applyHistoryFilters();
    });
    els.historyTags.appendChild(button);
  });
}

function renderHistoryList(items) {
  if (!els.historyList) return;

  els.historyList.innerHTML = "";

  if (!Array.isArray(items) || !items.length) {
    const li = document.createElement("li");
    li.className = "history-empty-shell";

    const emptyCard = document.createElement("article");
    emptyCard.className = "history-empty-card";
    emptyCard.setAttribute("data-reveal", "");

    const illustration = document.createElement("div");
    illustration.className = "history-empty-illustration";
    illustration.setAttribute("aria-hidden", "true");
    illustration.innerHTML = `
      <svg viewBox="0 0 120 120" fill="none">
        <rect x="22" y="28" width="76" height="60" rx="14" fill="rgba(212,161,61,0.12)"></rect>
        <path d="M32 41C32 37.69 34.69 35 38 35H54C59.22 35 64.18 37.25 67.6 41.17L68 41.63L68.4 41.17C71.82 37.25 76.78 35 82 35H88C91.31 35 94 37.69 94 41V81C94 83.21 92.21 85 90 85H82C76.78 85 71.82 87.25 68.4 91.17L68 91.63L67.6 91.17C64.18 87.25 59.22 85 54 85H36C33.79 85 32 83.21 32 81V41Z" stroke="currentColor" stroke-width="4" stroke-linejoin="round"></path>
        <path d="M68 43V88" stroke="currentColor" stroke-width="4" stroke-linecap="round"></path>
        <circle cx="92" cy="31" r="8" fill="rgba(31,75,57,0.18)"></circle>
      </svg>
    `;

    const copy = document.createElement("div");
    copy.className = "history-empty-copy";

    const hasSavedHistory = state.historyItems.length > 0;
    const heading = document.createElement("h3");
    heading.textContent = hasSavedHistory
      ? "No matching history right now."
      : "No history yet. Start by generating your first notes.";

    const message = document.createElement("p");
    message.textContent = hasSavedHistory
      ? "Try clearing your current search or filters to revisit previous lecture notes."
      : "Your generated lectures will appear here with subject tags, revisit actions, and faster access.";

    copy.appendChild(heading);
    copy.appendChild(message);

    const actionRow = document.createElement("div");
    actionRow.className = "history-empty-actions";

    if (hasSavedHistory) {
      const clearButton = document.createElement("button");
      clearButton.type = "button";
      clearButton.className = "ghost-btn";
      clearButton.textContent = "Clear Filters";
      clearButton.addEventListener("click", () => {
        state.historySearch = "";
        state.historyFilter = "all";
        state.historyTag = "all";
        if (els.historySearch) els.historySearch.value = "";
        if (els.historyFilter) els.historyFilter.value = "all";
        renderHistoryTags(state.historyItems);
        applyHistoryFilters();
      });
      actionRow.appendChild(clearButton);
    } else {
      const generateButton = document.createElement("button");
      generateButton.type = "button";
      generateButton.className = "primary-btn";
      generateButton.textContent = "Generate Notes";
      generateButton.addEventListener("click", () => {
        focusGenerateWorkspace();
      });
      actionRow.appendChild(generateButton);
    }

    emptyCard.appendChild(illustration);
    emptyCard.appendChild(copy);
    emptyCard.appendChild(actionRow);
    li.appendChild(emptyCard);
    els.historyList.appendChild(li);
    revealElements([emptyCard]);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.className = "history-item";

    const card = document.createElement("article");
    card.className = "history-card";
    card.setAttribute("data-reveal", "");

    const top = document.createElement("div");
    top.className = "history-card-top";

    const tag = document.createElement("span");
    tag.className = "history-subject";
    tag.textContent = `${getSubjectShortCode(item.subject)} | ${item.subject || "General"}`;

    const dateLabel = document.createElement("small");
    dateLabel.textContent = item.createdLabel || "-";

    top.appendChild(tag);
    top.appendChild(dateLabel);

    const title = document.createElement("h3");
    title.textContent = item.title || "Untitled video";

    const meta = document.createElement("p");
    meta.className = "history-meta";
    meta.textContent = item.videoUrl || "Saved lecture session";

    let preview = null;
    if (item.thumbnailUrl) {
      const previewLink = document.createElement(item.videoUrl ? "a" : "div");
      previewLink.className = "history-thumbnail";
      if (item.videoUrl) {
        previewLink.href = item.videoUrl;
        previewLink.target = "_blank";
        previewLink.rel = "noreferrer noopener";
      }

      const image = document.createElement("img");
      image.src = item.thumbnailUrl;
      image.alt = `${item.title || "Lecture"} thumbnail`;
      image.loading = "lazy";
      image.decoding = "async";

      const label = document.createElement("span");
      label.className = "history-thumbnail-label";
      label.textContent = "Lecture preview";

      previewLink.appendChild(image);
      previewLink.appendChild(label);
      preview = previewLink;
    }

    const actions = document.createElement("div");
    actions.className = "history-actions";

    const actionHint = document.createElement("small");
    actionHint.textContent = "Open, regenerate, or remove this card from your browser view";

    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.className = "history-open-btn";
    openButton.textContent = "Open";
    openButton.addEventListener("click", () => loadHistoryItem(item.id));

    const regenerateButton = document.createElement("button");
    regenerateButton.type = "button";
    regenerateButton.className = "history-open-btn history-regenerate-btn";
    regenerateButton.textContent = "Regenerate";
    regenerateButton.addEventListener("click", () => {
      if (!item.videoUrl) {
        setStatus("Saved lecture URL is unavailable for regeneration.", true);
        return;
      }

      if (els.youtubeUrl) {
        els.youtubeUrl.value = item.videoUrl;
      }

      focusGenerateWorkspace();
      generateNotes(item.videoUrl);
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "history-open-btn history-delete-btn";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", () => {
      hideHistoryItem(item.id);
      renderHistory(state.historyItems.filter((entry) => String(entry.id) !== String(item.id)));
      setStatus("History card removed from this browser view.");
    });

    actions.appendChild(actionHint);
    actions.appendChild(openButton);
    actions.appendChild(regenerateButton);
    actions.appendChild(deleteButton);

    card.appendChild(top);
    card.appendChild(title);
    card.appendChild(meta);
    if (preview) card.appendChild(preview);
    card.appendChild(actions);
    li.appendChild(card);
    els.historyList.appendChild(li);
    revealElements([card]);
  });
}

function applyHistoryFilters() {
  const filtered = state.historyItems.filter((item) => {
    if (state.historyTag !== "all" && item.subject !== state.historyTag) return false;
    if (state.historyFilter !== "all" && item.dateBucket !== state.historyFilter) return false;
    if (state.historySearch && !item.searchText.includes(state.historySearch)) return false;
    return true;
  });

  if (els.historyCount) {
    els.historyCount.textContent = `${filtered.length} session${filtered.length === 1 ? "" : "s"}`;
  }

  renderHistoryList(filtered);
}

function openDemoModal() {
  if (!els.demoModal) return;
  els.demoModal.classList.remove("hidden");
  els.demoModal.setAttribute("aria-hidden", "false");
}

function closeDemoModal() {
  if (!els.demoModal) return;
  els.demoModal.classList.add("hidden");
  els.demoModal.setAttribute("aria-hidden", "true");
}

function persistFeedbackDraft() {
  const draft = {
    name: String(els.feedbackName?.value || "").trim(),
    email: String(els.feedbackEmail?.value || "").trim(),
    message: String(els.feedbackMessage?.value || "").trim(),
  };
  localStorage.setItem("studykit_feedback_draft", JSON.stringify(draft));
}

function restoreFeedbackDraft() {
  const raw = localStorage.getItem("studykit_feedback_draft");
  if (!raw) return;

  try {
    const draft = JSON.parse(raw);
    if (els.feedbackName && draft?.name) els.feedbackName.value = draft.name;
    if (els.feedbackEmail && draft?.email) els.feedbackEmail.value = draft.email;
    if (els.feedbackMessage && draft?.message) els.feedbackMessage.value = draft.message;
  } catch (error) {
    localStorage.removeItem("studykit_feedback_draft");
  }
}

function saveHiddenHistoryIds() {
  localStorage.setItem(
    "studykit_hidden_history_ids",
    JSON.stringify(Array.from(state.hiddenHistoryIds))
  );
}

function hideHistoryItem(id) {
  if (!id) return;
  state.hiddenHistoryIds.add(String(id));
  saveHiddenHistoryIds();
}

function updateStudentProgress() {
  const visibleSessions = state.historyItems.length;
  const progress = Math.min(100, visibleSessions * 12);
  const lastActive = state.currentResult ? "Last active in this study session" : "Last active just now";

  if (els.studentProgressLabel) {
    els.studentProgressLabel.textContent = `${progress}%`;
  }

  if (els.studentProgressBar) {
    els.studentProgressBar.style.width = `${progress}%`;
  }

  if (els.studentLastActive) {
    els.studentLastActive.textContent = lastActive;
  }
}

function renderStudentAvatar(name) {
  const safeName = String(name || "").trim();
  const initials = safeName
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() || "")
    .join("") || "ST";

  if (els.studentAvatar) {
    els.studentAvatar.textContent = initials;
  }

  if (els.studentProfileName) {
    els.studentProfileName.textContent = safeName || "StudyKit Student";
  }
}

function startHeroTyping() {
  if (!els.heroTypingTarget) return;

  const phrases = state.heroTypingPhrases;
  let phraseIndex = 0;
  let charIndex = 0;
  let deleting = false;

  const tick = () => {
    const phrase = phrases[phraseIndex] || "";

    if (!deleting) {
      charIndex += 1;
      els.heroTypingTarget.textContent = phrase.slice(0, charIndex);
      if (charIndex >= phrase.length) {
        deleting = true;
        state.heroTypingTimer = window.setTimeout(tick, 1200);
        return;
      }
      state.heroTypingTimer = window.setTimeout(tick, 55);
      return;
    }

    charIndex -= 1;
    els.heroTypingTarget.textContent = phrase.slice(0, Math.max(0, charIndex));
    if (charIndex <= 0) {
      deleting = false;
      phraseIndex = (phraseIndex + 1) % phrases.length;
      state.heroTypingTimer = window.setTimeout(tick, 260);
      return;
    }

    state.heroTypingTimer = window.setTimeout(tick, 28);
  };

  els.heroTypingTarget.textContent = "";
  tick();
}

function revealElements(elements) {
  const safeElements = Array.from(elements || []).filter((element) => element instanceof HTMLElement);
  if (!safeElements.length) return;

  if (!("IntersectionObserver" in window)) {
    safeElements.forEach((element) => element.classList.add("is-visible"));
    return;
  }

  if (!state.revealObserver) {
    state.revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          state.revealObserver?.unobserve(entry.target);
        });
      },
      {
        threshold: 0.14,
        rootMargin: "0px 0px -8% 0px",
      }
    );
  }

  safeElements.forEach((element) => {
    if (element.classList.contains("is-visible")) return;
    state.revealObserver.observe(element);
  });
}

function initializeRevealAnimations() {
  revealElements(document.querySelectorAll("[data-reveal]"));
}

function getSubjectShortCode(subject) {
  const map = {
    Physics: "PHY",
    Chemistry: "CHE",
    Mathematics: "MTH",
    Biology: "BIO",
    "Computer Science": "CSE",
    History: "HIS",
    Economics: "ECO",
    English: "ENG",
    General: "GEN",
  };
  return map[subject] || "GEN";
}

function setTheme(theme) {
  const normalized = theme === "dark" ? "dark" : "light";
  document.body.dataset.theme = normalized;
  localStorage.setItem("theme", normalized);

  if (els.themeToggle) {
    els.themeToggle.textContent = normalized === "dark" ? "Use Light Theme" : "Use Dark Theme";
  }

  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: false,
      theme: normalized === "dark" ? "dark" : "default",
      securityLevel: "loose",
    });
  }
}

function getSavedTheme() {
  return localStorage.getItem("theme") || "light";
}

function setAuthMode(mode) {
  state.authMode = mode === "register" ? "register" : "login";
  if (els.authTabLogin) els.authTabLogin.classList.toggle("active", state.authMode === "login");
  if (els.authTabRegister) els.authTabRegister.classList.toggle("active", state.authMode === "register");
  if (els.loginForm) els.loginForm.classList.toggle("hidden", state.authMode !== "login");
  if (els.registerForm) els.registerForm.classList.toggle("hidden", state.authMode !== "register");

  if (state.authMode === "login") {
    setAuthStatus("Login to access your student workspace.");
  } else {
    setAuthStatus("Create your student account to start using StudyKit Pro.");
  }
}

function renderStudentProfile(student) {
  if (!student) return;
  if (els.studentName) els.studentName.textContent = student.full_name || "-";
  if (els.studentMobile) els.studentMobile.textContent = student.mobile_number || "-";
  if (els.studentCreated) els.studentCreated.textContent = formatIsoDate(student.created_at);
  if (els.sessionPill) els.sessionPill.textContent = `Student: ${student.full_name || "Logged In"}`;
  renderStudentAvatar(student.full_name || "");
  updateStudentProgress();
  if (els.feedbackName && !els.feedbackName.value) {
    els.feedbackName.value = student.full_name || "";
  }
}

function renderStudentDirectory(items) {
  if (!els.studentList) return;
  els.studentList.innerHTML = "";

  if (!Array.isArray(items) || !items.length) {
    const li = document.createElement("li");
    li.className = "student-item";
    li.textContent = "No students found.";
    els.studentList.appendChild(li);
    return;
  }

  items.forEach((student) => {
    const li = document.createElement("li");
    li.className = "student-item";

    const name = document.createElement("strong");
    name.textContent = student.full_name || "Unnamed Student";

    const meta = document.createElement("small");
    meta.textContent = `${student.mobile_number || "-"} | Joined: ${formatIsoDate(student.created_at)}`;

    li.appendChild(name);
    li.appendChild(meta);
    els.studentList.appendChild(li);
  });
}

function clearNotesUi() {
  state.sourceResult = null;
  state.currentResult = null;
  state.activeLanguage = "en";
  state.translationCache = {};
  state.historyItems = [];
  state.historySearch = "";
  state.historyFilter = "all";
  state.historyTag = "all";

  if (els.historyList) els.historyList.innerHTML = "";
  if (els.languageSelect) els.languageSelect.value = "en";
  if (els.historySearch) els.historySearch.value = "";
  if (els.historyFilter) els.historyFilter.value = "all";
  if (els.historyTags) els.historyTags.innerHTML = "";
  if (els.historyCount) els.historyCount.textContent = "0 sessions";
  if (els.studentProfileName) els.studentProfileName.textContent = "Ready to learn";
  if (els.studentLastActive) els.studentLastActive.textContent = "Last active just now";
  if (els.studentAvatar) els.studentAvatar.textContent = "ST";
  if (els.studentProgressLabel) els.studentProgressLabel.textContent = "0%";
  if (els.studentProgressBar) els.studentProgressBar.style.width = "0%";

  setLanguageEnabled(false);
  setExportEnabled(false);
  updateLanguageStatus();
  updateDashboardStats([]);
  renderNotesEmptyState();
}

function applyAuthUi() {
  const isLoggedIn = Boolean(state.authToken && state.currentStudent);

  if (els.authSection) els.authSection.classList.toggle("hidden", isLoggedIn);
  if (els.studentSection) els.studentSection.classList.toggle("hidden", !isLoggedIn);
  if (els.secureApp) els.secureApp.classList.toggle("hidden", !isLoggedIn);
  if (els.logoutBtn) els.logoutBtn.classList.toggle("hidden", !isLoggedIn);
  if (els.sessionPill) els.sessionPill.classList.toggle("hidden", !isLoggedIn);

  if (!isLoggedIn) {
    clearNotesUi();
    setStatus("Login to generate notes.");
    return;
  }

  initializeRevealAnimations();

  if (!state.currentResult) {
    renderNotesEmptyState();
  }
}

function authHeaders() {
  if (!state.authToken) return {};
  return { Authorization: `Bearer ${state.authToken}` };
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...authHeaders(),
    },
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed.");
  }

  return response.json();
}

function saveSession(token, student) {
  state.authToken = String(token || "");
  state.currentStudent = student || null;

  if (state.authToken) localStorage.setItem("studykit_auth_token", state.authToken);
  else localStorage.removeItem("studykit_auth_token");

  renderStudentProfile(student);
  applyAuthUi();
}

function clearSession() {
  state.authToken = "";
  state.currentStudent = null;
  localStorage.removeItem("studykit_auth_token");
  renderStudentDirectory([]);
  applyAuthUi();
}

function ensureAuthenticated() {
  if (state.authToken && state.currentStudent) return true;
  setAuthStatus("Login first to access your workspace.", true);
  return false;
}

async function loadStudentDirectory() {
  try {
    const students = await apiRequest("/api/students?limit=50");
    renderStudentDirectory(students);
  } catch (error) {
    renderStudentDirectory([]);
    setAuthStatus(normalizeErrorMessage(error.message), true);
  }
}

async function restoreSession() {
  if (!state.authToken) return;

  try {
    const me = await apiRequest("/api/auth/me");
    saveSession(state.authToken, me);
    await loadStudentDirectory();
    await loadHistory();
    setStatus("Welcome back. Paste a lecture URL to begin.");
  } catch (error) {
    clearSession();
    setAuthStatus("Session expired. Please login again.", true);
  }
}

async function handleLoginSubmit(event) {
  event.preventDefault();

  const mobileNumber = normalizeMobileNumber(
    String(document.getElementById("login-mobile")?.value || "")
  );
  const password = String(document.getElementById("login-password")?.value || "");

  if (!mobileNumber || !password) {
    setAuthStatus("Mobile number and password are required.", true);
    return;
  }

  setAuthBusy(true, "login");
  setLoading(true);
  startLoadingSequence(["Verifying student...", "Opening workspace...", "Syncing saved sessions..."]);
  setAuthStatus("Signing in...");

  try {
    const payload = await apiRequest("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mobile_number: mobileNumber, password }),
    });

    saveSession(payload.token, payload.student);
    await loadStudentDirectory();
    await loadHistory();
    setAuthStatus(`Welcome ${payload.student.full_name}.`);
    setStatus("Paste a lecture URL to begin.");
    if (els.loginForm) els.loginForm.reset();
  } catch (error) {
    setAuthStatus(normalizeErrorMessage(error.message), true);
  } finally {
    setAuthBusy(false, "login");
    setLoading(false);
  }
}

function normalizeMobileNumber(value) {
  const raw = String(value || "").trim();
  const digits = raw.replace(/\D/g, "");
  if (digits.length < 10 || digits.length > 15) return "";
  if (raw.startsWith("+")) return `+${digits}`;
  if (digits.length === 10) return digits;
  return `+${digits}`;
}

async function handleRegisterSubmit(event) {
  event.preventDefault();

  const fullName = String(document.getElementById("register-name")?.value || "").trim();
  const mobileNumber = normalizeMobileNumber(
    String(document.getElementById("register-mobile")?.value || "")
  );
  const password = String(document.getElementById("register-password")?.value || "");
  const confirmPassword = String(document.getElementById("register-confirm-password")?.value || "");

  if (!fullName || !mobileNumber || !password) {
    setAuthStatus("Please fill all required registration fields.", true);
    return;
  }
  if (password.length < 8) {
    setAuthStatus("Password must be at least 8 characters.", true);
    return;
  }
  if (password !== confirmPassword) {
    setAuthStatus("Password and confirm password do not match.", true);
    return;
  }

  setAuthBusy(true, "register");
  setLoading(true);
  startLoadingSequence(["Creating student account...", "Preparing secure workspace...", "Saving profile details..."]);
  setAuthStatus("Creating account...");

  try {
    const payload = await apiRequest("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        full_name: fullName,
        mobile_number: mobileNumber,
        password,
      }),
    });

    saveSession(payload.token, payload.student);
    await loadStudentDirectory();
    await loadHistory();
    setAuthStatus(`Account created for ${payload.student.full_name}.`);
    setStatus("Paste a lecture URL to begin.");
    if (els.registerForm) els.registerForm.reset();
  } catch (error) {
    setAuthStatus(normalizeErrorMessage(error.message), true);
  } finally {
    setAuthBusy(false, "register");
    setLoading(false);
  }
}

function normalizeStringList(value) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item || "").trim()).filter((item) => item.length > 0);
  }
  if (value === null || value === undefined) return [];
  const text = String(value).trim();
  return text ? [text] : [];
}

function getTopicKeyTerms(topic) {
  const raw = Array.isArray(topic.key_terms) ? topic.key_terms : (Array.isArray(topic.glossary) ? topic.glossary : []);

  return raw
    .map((item) => {
      if (item && typeof item === "object") {
        return {
          term: String(item.term || item.name || "").trim(),
          definition: String(item.definition || item.meaning || item.explanation || "").trim(),
        };
      }

      const text = String(item || "").trim();
      if (!text) return { term: "", definition: "" };
      if (text.includes(":")) {
        const [head, ...tail] = text.split(":");
        return { term: head.trim(), definition: tail.join(":").trim() };
      }
      return { term: text, definition: "" };
    })
    .filter((item) => item.term.length > 0);
}

function getTopicSelfCheck(topic) {
  const raw = Array.isArray(topic.self_check) ? topic.self_check : (Array.isArray(topic.practice) ? topic.practice : []);

  return raw
    .map((item) => {
      if (item && typeof item === "object") {
        return {
          question: String(item.question || "").trim(),
          answer: String(item.answer || item.expected_answer || "").trim(),
        };
      }
      return { question: String(item || "").trim(), answer: "" };
    })
    .filter((item) => item.question.length > 0);
}

function getTopicDiagrams(topic) {
  const diagrams = Array.isArray(topic.diagrams) ? topic.diagrams : [];
  const normalized = diagrams
    .filter((item) => item && typeof item === "object")
    .map((item) => ({
      title: String(item.title || "Diagram").trim() || "Diagram",
      diagramType: String(item.diagram_type || "mermaid").trim() || "mermaid",
      mermaid: String(item.mermaid || "").trim(),
    }))
    .filter((item) => item.mermaid.length > 0);

  if (normalized.length) return normalized;

  const fallback = String(topic.diagram || "").trim();
  if (!fallback) return [];

  return [{ title: "Diagram", diagramType: "mermaid", mermaid: fallback }];
}

function createListSection(title) {
  const section = document.createElement("section");
  section.className = "topic-block";
  const heading = document.createElement("h5");
  heading.textContent = title;
  section.appendChild(heading);
  return section;
}

function appendBulletList(section, values) {
  if (!values.length) return;
  const list = document.createElement("ul");
  list.className = "topic-list";
  values.forEach((value) => {
    const li = document.createElement("li");
    li.textContent = value;
    list.appendChild(li);
  });
  section.appendChild(list);
}

function renderConceptBadges(card, terms) {
  if (!Array.isArray(terms) || !terms.length) return;

  const wrap = document.createElement("div");
  wrap.className = "concept-badges";

  terms.slice(0, 6).forEach((item) => {
    const badge = document.createElement("span");
    badge.className = "concept-badge";
    badge.textContent = item.term || "Concept";
    wrap.appendChild(badge);
  });

  card.appendChild(wrap);
}

function appendKeyNoteCards(section, values) {
  if (!Array.isArray(values) || !values.length) return;

  const wrap = document.createElement("div");
  wrap.className = "highlight-grid";

  values.forEach((value) => {
    const note = document.createElement("article");
    const text = String(value || "").trim();
    const isImportant = /important|remember|exam|must|key/i.test(text);

    note.className = `highlight-card${isImportant ? " important" : ""}`;

    const label = document.createElement("p");
    label.className = "highlight-label";
    label.textContent = isImportant ? "Important" : "Key Point";

    const body = document.createElement("p");
    body.className = "highlight-text";
    body.textContent = text;

    note.appendChild(label);
    note.appendChild(body);
    wrap.appendChild(note);
  });

  section.appendChild(wrap);
}

function appendKeyTermCards(section, terms) {
  if (!Array.isArray(terms) || !terms.length) return;

  const wrap = document.createElement("div");
  wrap.className = "definition-grid";

  terms.forEach((item) => {
    const card = document.createElement("article");
    card.className = "definition-card";

    const term = document.createElement("strong");
    term.className = "definition-term";
    term.textContent = item.term || "Key term";

    const definition = document.createElement("p");
    definition.className = "definition-copy";
    definition.textContent = item.definition || "Definition not provided.";

    card.appendChild(term);
    card.appendChild(definition);
    wrap.appendChild(card);
  });

  section.appendChild(wrap);
}

function appendSelfCheckCards(section, values) {
  if (!Array.isArray(values) || !values.length) return;

  const wrap = document.createElement("div");
  wrap.className = "self-check-grid";

  values.forEach((item) => {
    const card = document.createElement("article");
    card.className = "self-check-card";

    const question = document.createElement("strong");
    question.className = "self-check-question";
    question.textContent = item.question || "Question";

    const answer = document.createElement("p");
    answer.className = "self-check-answer";
    answer.textContent = item.answer ? `Answer guide: ${item.answer}` : "Answer guide not available.";

    card.appendChild(question);
    card.appendChild(answer);
    wrap.appendChild(card);
  });

  section.appendChild(wrap);
}

function splitExplanationIntoChunks(value) {
  const text = String(value || "").trim();
  if (!text) return [];

  const explicitParts = text
    .split(/\n+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);

  if (explicitParts.length > 1) return explicitParts;

  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);

  if (sentences.length > 1) return sentences;

  const softParts = text
    .split(/\s*(?:;|,\s+(?=[A-Z])|\n)\s*/g)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);

  if (softParts.length > 1) return softParts;

  return [text];
}

function normalizeQuizValue(value) {
  return String(value || "")
    .trim()
    .replace(/^[A-D][).:\-\s]+/i, "")
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function getQuizCorrectIndex(item) {
  const options = Array.isArray(item?.options) ? item.options.map((option) => String(option || "")) : [];
  const rawAnswer = String(item?.correct_answer || "").trim();
  if (!options.length || !rawAnswer) return -1;

  const answerMatch = rawAnswer.match(/^([A-D])(?:[\).:\-\s]|$)/i);
  if (answerMatch) {
    const index = "ABCD".indexOf(answerMatch[1].toUpperCase());
    if (index >= 0 && index < options.length) return index;
  }

  const normalizedAnswer = normalizeQuizValue(rawAnswer);
  return options.findIndex((option) => normalizeQuizValue(option) === normalizedAnswer);
}

function getQuizCorrectLabel(item, correctIndex) {
  const options = Array.isArray(item?.options) ? item.options : [];
  if (correctIndex >= 0 && correctIndex < options.length) return String(options[correctIndex] || "").trim();
  return String(item?.correct_answer || "").trim();
}

function looksLikeMermaidSource(source) {
  return /flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram|journey|gantt|pie|mindmap|timeline|quadrantChart|requirementDiagram/i.test(String(source || "").trim());
}

async function renderDiagram(container, text, key) {
  const source = String(text || "").trim();
  if (!source) {
    container.textContent = "No diagram provided.";
    return;
  }

  const looksLikeMermaid = looksLikeMermaidSource(source);

  if (!window.mermaid || !looksLikeMermaid) {
    const pre = document.createElement("pre");
    pre.textContent = source;
    container.appendChild(pre);
    return;
  }

  try {
    const { svg } = await mermaid.render(`diagram-${key}-${Date.now()}`, source);
    container.innerHTML = svg;
  } catch (error) {
    const pre = document.createElement("pre");
    pre.textContent = source;
    container.appendChild(pre);
  }
}

async function renderTopicDiagrams(container, topic, topicIndex) {
  const diagrams = getTopicDiagrams(topic);

  if (!diagrams.length) {
    const empty = document.createElement("div");
    empty.className = "diagram-box";
    empty.textContent = "No diagram provided.";
    container.appendChild(empty);
    return;
  }

  for (let index = 0; index < diagrams.length; index += 1) {
    const diagram = diagrams[index];
    const shell = document.createElement("section");
    shell.className = "diagram-shell";

    const title = document.createElement("p");
    title.className = "diagram-title";
    title.textContent = `${diagram.title} (${diagram.diagramType})`;

    const body = document.createElement("div");
    body.className = "diagram-box";

    shell.appendChild(title);
    shell.appendChild(body);
    container.appendChild(shell);

    await renderDiagram(body, diagram.mermaid, `${topicIndex}-${index}`);
  }
}

async function renderResult(data) {
  els.videoTitle.textContent = data.video_title || "Untitled video";

  const videoUrl = String(data.video_url || "").trim();
  if (videoUrl) {
    els.videoLink.href = videoUrl;
    els.videoLink.textContent = videoUrl;
    els.videoLink.classList.remove("disabled-link");
  } else {
    els.videoLink.href = "#";
    els.videoLink.textContent = "Generate notes to view video details";
    els.videoLink.classList.add("disabled-link");
  }

  const topics = Array.isArray(data.topics) ? data.topics : [];
  setExportEnabled(Boolean(topics.length));
  setNotesContentVisibility(Boolean(topics.length));

  els.tocList.innerHTML = "";
  if (!topics.length) {
    renderNotesEmptyState();
    return;
  }

  topics.forEach((topic, index) => {
    const li = document.createElement("li");
    li.className = "toc-item";
    const link = document.createElement("a");
    link.className = "toc-link";
    link.href = `#topic-${index}`;
    link.textContent = `${index + 1}. ${topic.title || `Topic ${index + 1}`}`;
    li.appendChild(link);
    els.tocList.appendChild(li);
  });

  els.topicCards.innerHTML = "";

  for (let index = 0; index < topics.length; index += 1) {
    const topic = topics[index];

    const card = document.createElement("article");
    card.className = "topic-card";
    card.id = `topic-${index}`;

    const top = document.createElement("div");
    top.className = "topic-top";
    const tag = document.createElement("p");
    tag.className = "topic-index";
    tag.textContent = `Topic ${index + 1}`;
    const heading = document.createElement("h4");
    heading.textContent = topic.title || `Topic ${index + 1}`;
    top.appendChild(tag);
    top.appendChild(heading);
    card.appendChild(top);

    const terms = getTopicKeyTerms(topic);
    renderConceptBadges(card, terms);

    const explanationSection = createListSection("Explanation");
    const explanationStack = document.createElement("div");
    explanationStack.className = "explanation-stack";

    const explanationChunks = splitExplanationIntoChunks(topic.explanation);
    if (!explanationChunks.length) {
      const explanationItem = document.createElement("p");
      explanationItem.className = "explanation-item";
      explanationItem.textContent = "No explanation provided.";
      explanationStack.appendChild(explanationItem);
    } else {
      explanationChunks.forEach((chunk) => {
        const explanationItem = document.createElement("p");
        explanationItem.className = "explanation-item";
        explanationItem.textContent = chunk;
        explanationStack.appendChild(explanationItem);
      });
    }

    explanationSection.appendChild(explanationStack);
    card.appendChild(explanationSection);

    const notes = normalizeStringList(topic.bullet_notes);
    if (notes.length) {
      const noteSection = createListSection("Key Notes");
      appendKeyNoteCards(noteSection, notes);
      card.appendChild(noteSection);
    }

    if (terms.length) {
      const termSection = createListSection("Key Terms");
      appendKeyTermCards(termSection, terms);
      card.appendChild(termSection);
    }

    const selfCheck = getTopicSelfCheck(topic);
    if (selfCheck.length) {
      const checkSection = createListSection("Self Check");
      appendSelfCheckCards(checkSection, selfCheck);
      card.appendChild(checkSection);
    }

    const diagramsSection = createListSection("Diagrams");
    const diagramStack = document.createElement("div");
    diagramStack.className = "diagram-stack";
    diagramsSection.appendChild(diagramStack);
    card.appendChild(diagramsSection);

    const quiz = Array.isArray(topic.quiz) ? topic.quiz : [];
    if (quiz.length) {
      const quizSection = createListSection("Quiz Checkpoints");
      const quizList = document.createElement("div");
      quizList.className = "quiz-list";

      quiz.forEach((item, qIndex) => {
        const quizItem = document.createElement("div");
        quizItem.className = "quiz-item";

        const qTitle = document.createElement("h4");
        qTitle.textContent = `${qIndex + 1}. ${item.question || "Question"}`;

        const options = document.createElement("div");
        options.className = "quiz-options";

        const correctIndex = getQuizCorrectIndex(item);
        const feedback = document.createElement("p");
        feedback.className = "quiz-feedback";
        feedback.textContent = "Choose one answer to check yourself.";

        const optionButtons = [];
        (item.options || []).forEach((option, optionIndex) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "quiz-option";
          button.textContent = option;
          button.addEventListener("click", () => {
            optionButtons.forEach((entry) => {
              entry.classList.remove("selected", "correct", "wrong");
            });

            button.classList.add("selected");

            if (correctIndex === -1) {
              feedback.className = "quiz-feedback";
              feedback.textContent = "Answer key unavailable for this question.";
              return;
            }

            if (optionIndex === correctIndex) {
              button.classList.add("correct");
              feedback.className = "quiz-feedback correct";
              feedback.textContent = "Correct answer.";
              return;
            }

            button.classList.add("wrong");
            optionButtons[correctIndex]?.classList.add("correct");
            feedback.className = "quiz-feedback wrong";
            feedback.textContent = `Wrong answer. Correct answer: ${getQuizCorrectLabel(item, correctIndex)}`;
          });

          optionButtons.push(button);
          options.appendChild(button);
        });

        quizItem.appendChild(qTitle);
        quizItem.appendChild(options);
        quizItem.appendChild(feedback);
        quizList.appendChild(quizItem);
      });

      quizSection.appendChild(quizList);
      card.appendChild(quizSection);
    }

    els.topicCards.appendChild(card);
    await renderTopicDiagrams(diagramStack, topic, index);
  }
}

function initializeResultState(result) {
  state.sourceResult = result;
  state.currentResult = result;
  state.activeLanguage = "en";
  state.translationCache = { en: result };
  if (els.languageSelect) els.languageSelect.value = "en";
  updateLanguageStatus();
  setLanguageEnabled(true);
  setExportEnabled(Boolean(result && Array.isArray(result.topics) && result.topics.length));
  updateStudentProgress();
}

function sanitizeFileName(value) {
  return String(value || "study-notes")
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80) || "study-notes";
}

async function svgMarkupToPngDataUrl(svgMarkup) {
  const blob = new Blob([svgMarkup], { type: "image/svg+xml;charset=utf-8" });
  const objectUrl = URL.createObjectURL(blob);

  try {
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = objectUrl;
    });

    const canvas = document.createElement("canvas");
    const width = Math.max(1200, image.naturalWidth || 1200);
    const height = Math.max(700, image.naturalHeight || 700);
    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas context unavailable.");

    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    return canvas.toDataURL("image/png");
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function renderDiagramToPngDataUrl(text, key) {
  const source = String(text || "").trim();
  if (!source || !window.mermaid || !looksLikeMermaidSource(source)) return null;

  try {
    const { svg } = await mermaid.render(`pdf-diagram-${key}-${Date.now()}`, source);
    return await svgMarkupToPngDataUrl(svg);
  } catch (error) {
    return null;
  }
}

async function exportNotesToPdf() {
  if (!state.currentResult) {
    setStatus("Generate notes first, then download the PDF.", true);
    return;
  }

  if (!window.jspdf?.jsPDF) {
    setStatus("PDF export library failed to load.", true);
    return;
  }

  setLoading(true);
  setGenerateBusy(true);
  setExportEnabled(false);
  startLoadingSequence(["Preparing PDF layout...", "Embedding diagrams...", "Finalizing export..."]);
  setButtonLoading(els.downloadPdfBtn, true, "Preparing PDF...");
  setStatus("Preparing PDF...");

  try {
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF("p", "mm", "a4");
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 12;
    const contentWidth = pageWidth - margin * 2;
    const bottomLimit = pageHeight - margin;
    let y = margin;

    const ensureSpace = (needed = 8) => {
      if (y + needed <= bottomLimit) return;
      pdf.addPage();
      y = margin;
    };

    const addWrappedText = (text, fontSize = 11, lineHeight = 5.6, color = [85, 101, 122]) => {
      const safeText = String(text || "").trim();
      if (!safeText) return;
      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(fontSize);
      pdf.setTextColor(color[0], color[1], color[2]);
      const lines = pdf.splitTextToSize(safeText, contentWidth);
      lines.forEach((line) => {
        ensureSpace(lineHeight);
        pdf.text(line, margin, y);
        y += lineHeight;
      });
    };

    const addHeading = (text, size = 16) => {
      ensureSpace(size === 20 ? 12 : 10);
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(size);
      pdf.setTextColor(15, 27, 45);
      pdf.text(String(text || ""), margin, y);
      y += size === 20 ? 10 : 8;
    };

    const addSectionLabel = (text) => {
      ensureSpace(8);
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(10);
      pdf.setTextColor(11, 99, 204);
      pdf.text(String(text || "").toUpperCase(), margin, y);
      y += 6;
    };

    const addBulletList = (items) => {
      const safeItems = Array.isArray(items) ? items : [];
      safeItems.forEach((item) => {
        const lines = pdf.splitTextToSize(`- ${String(item || "").trim()}`, contentWidth - 2);
        lines.forEach((line) => {
          ensureSpace(5.2);
          pdf.setFont("helvetica", "normal");
          pdf.setFontSize(10.5);
          pdf.setTextColor(40, 53, 74);
          pdf.text(line, margin + 1, y);
          y += 5.2;
        });
      });
    };

    const addDivider = () => {
      ensureSpace(6);
      pdf.setDrawColor(214, 226, 240);
      pdf.line(margin, y, pageWidth - margin, y);
      y += 6;
    };

    const videoTitle = state.currentResult.video_title || "Study Notes";
    const videoUrl = String(state.currentResult.video_url || "").trim();
    const languageLabel = LANGUAGE_LABELS[state.activeLanguage] || state.activeLanguage.toUpperCase();
    const topics = Array.isArray(state.currentResult.topics) ? state.currentResult.topics : [];
    const roadmap = Array.isArray(state.currentResult.table_of_contents)
      ? state.currentResult.table_of_contents
      : topics.map((topic) => topic?.title || "Topic");

    addHeading(videoTitle, 20);
    addWrappedText(`Language: ${languageLabel}`, 10, 5);
    if (videoUrl) addWrappedText(`Source: ${videoUrl}`, 9.5, 5);
    y += 2;

    if (roadmap.length) {
      addSectionLabel("Learning Roadmap");
      addBulletList(roadmap);
      y += 4;
    }

    for (let topicIndex = 0; topicIndex < topics.length; topicIndex += 1) {
      const topic = topics[topicIndex] || {};
      addDivider();
      addHeading(`${topicIndex + 1}. ${topic.title || `Topic ${topicIndex + 1}`}`, 15);

      const explanationChunks = splitExplanationIntoChunks(topic.explanation);
      if (explanationChunks.length) {
        addSectionLabel("Explanation");
        explanationChunks.forEach((chunk) => {
          addWrappedText(chunk, 11, 6);
          y += 1;
        });
      }

      const notes = normalizeStringList(topic.bullet_notes);
      if (notes.length) {
        addSectionLabel("Key Notes");
        addBulletList(notes);
        y += 2;
      }

      const terms = getTopicKeyTerms(topic);
      if (terms.length) {
        addSectionLabel("Key Terms");
        addBulletList(terms.map((item) => (item.definition ? `${item.term}: ${item.definition}` : item.term)));
        y += 2;
      }

      const selfCheck = getTopicSelfCheck(topic);
      if (selfCheck.length) {
        addSectionLabel("Self Check");
        addBulletList(selfCheck.map((item) => (item.answer ? `${item.question} -> ${item.answer}` : item.question)));
        y += 2;
      }

      const diagrams = getTopicDiagrams(topic);
      if (diagrams.length) {
        addSectionLabel("Diagrams");

        for (let diagramIndex = 0; diagramIndex < diagrams.length; diagramIndex += 1) {
          const diagram = diagrams[diagramIndex];
          addWrappedText(`${diagram.title} (${diagram.diagramType})`, 10, 5.2, [40, 53, 74]);
          y += 1;

          const diagramImage = await renderDiagramToPngDataUrl(
            diagram.mermaid,
            `${topicIndex}-${diagramIndex}`
          );

          if (diagramImage) {
            const imageProps = pdf.getImageProperties(diagramImage);
            const maxDiagramWidth = contentWidth;
            const maxDiagramHeight = 90;
            let imageWidth = maxDiagramWidth;
            let imageHeight = (imageProps.height * imageWidth) / imageProps.width;

            if (imageHeight > maxDiagramHeight) {
              imageHeight = maxDiagramHeight;
              imageWidth = (imageProps.width * imageHeight) / imageProps.height;
            }

            ensureSpace(imageHeight + 4);
            pdf.addImage(diagramImage, "PNG", margin, y, imageWidth, imageHeight);
            y += imageHeight + 5;
          } else {
            addWrappedText(diagram.mermaid || "Diagram not available.", 9.5, 5);
            y += 2;
          }
        }
      }

      const quiz = Array.isArray(topic.quiz) ? topic.quiz : [];
      if (quiz.length) {
        addSectionLabel("Quiz Checkpoints");
        quiz.forEach((item, quizIndex) => {
          addWrappedText(`${quizIndex + 1}. ${item.question || "Question"}`, 10.5, 5.4, [15, 27, 45]);
          addBulletList((item.options || []).map((option) => String(option || "")));
          const correctIndex = getQuizCorrectIndex(item);
          addWrappedText(`Answer: ${getQuizCorrectLabel(item, correctIndex) || "Not provided"}`, 10, 5.2, [11, 99, 204]);
          y += 2;
        });
      }
    }

    const fileName = sanitizeFileName(`${state.currentResult.video_title || "study-notes"}-${languageLabel}`);
    pdf.save(`${fileName}.pdf`);
    setStatus("PDF downloaded successfully.");
  } catch (error) {
    setStatus(normalizeErrorMessage(error.message || "Failed to export PDF."), true);
  } finally {
    setLoading(false);
    setGenerateBusy(false);
    setExportEnabled(Boolean(state.currentResult));
    setButtonLoading(els.downloadPdfBtn, false);
  }
}

async function generateNotes(url) {
  if (!ensureAuthenticated()) return;

  setStatus("Generating notes...");
  setLoading(true);
  setGenerateBusy(true);
  setLanguageEnabled(false);
  startLoadingSequence(["Fetching lecture...", "Analyzing content...", "Generating notes..."]);
  showNotesSkeleton("Generating smart notes...");

  try {
    const data = await apiRequest("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    initializeResultState(data);
    await renderResult(data);
    setStatus("Notes generated successfully.");
    await loadHistory();
  } catch (error) {
    setStatus(normalizeErrorMessage(error.message), true);
    if (state.currentResult) {
      await renderResult(state.currentResult);
    } else {
      renderNotesEmptyState();
    }
  } finally {
    setLoading(false);
    setGenerateBusy(false);
    if (state.sourceResult) setLanguageEnabled(true);
  }
}

async function switchLanguage(nextLanguage) {
  if (!ensureAuthenticated()) return;

  if (!state.sourceResult) {
    setStatus("Generate notes first, then change language.", true);
    if (els.languageSelect) els.languageSelect.value = state.activeLanguage;
    return;
  }

  const selectedCode = String(nextLanguage || "en").toLowerCase();
  const selectedName = LANGUAGE_LABELS[selectedCode] || selectedCode.toUpperCase();

  if (selectedCode === state.activeLanguage) {
    updateLanguageStatus();
    return;
  }

  if (selectedCode === "en") {
    state.currentResult = state.sourceResult;
    state.activeLanguage = "en";
    await renderResult(state.currentResult);
    updateLanguageStatus();
    setStatus("Showing English notes.");
    return;
  }

  if (state.translationCache[selectedCode]) {
    state.currentResult = state.translationCache[selectedCode];
    state.activeLanguage = selectedCode;
    await renderResult(state.currentResult);
    updateLanguageStatus();
    setStatus(`Showing ${selectedName} notes.`);
    return;
  }

  setStatus(`Translating notes to ${selectedName}...`);
  setLanguageStatus(`Translating to ${selectedName}...`);
  setLoading(true);
  setGenerateBusy(true);
  setLanguageEnabled(false);
  startLoadingSequence([`Reading ${selectedName} context...`, "Rewriting note sections...", "Preparing translated study pack..."]);

  try {
    const translated = await apiRequest("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        notes: state.sourceResult,
        target_language: selectedCode,
        target_language_name: selectedName,
      }),
    });

    state.translationCache[selectedCode] = translated;
    state.currentResult = translated;
    state.activeLanguage = selectedCode;

    await renderResult(translated);
    updateLanguageStatus();
    setStatus(`Showing ${selectedName} notes.`);
  } catch (error) {
    setStatus(normalizeErrorMessage(error.message), true);
    setLanguageStatus(`Unable to translate to ${selectedName}.`, true);
    if (els.languageSelect) els.languageSelect.value = state.activeLanguage;
  } finally {
    setLoading(false);
    setGenerateBusy(false);
    setLanguageEnabled(Boolean(state.sourceResult));
  }
}

function renderHistory(items) {
  const enriched = Array.isArray(items) ? items.map(enrichHistoryItem) : [];
  state.historyItems = enriched.filter((item) => !state.hiddenHistoryIds.has(String(item.id)));
  updateDashboardStats(state.historyItems);
  updateStudentProgress();
  renderHistoryTags(state.historyItems);
  applyHistoryFilters();
}

async function loadHistory() {
  if (!ensureAuthenticated()) {
    renderHistory([]);
    return;
  }

  try {
    const data = await apiRequest("/api/history");
    renderHistory(data);
  } catch (error) {
    renderHistory([]);
  }
}

async function loadHistoryItem(id) {
  if (!id || !ensureAuthenticated()) return;

  setStatus("Loading saved notes...");
  setLoading(true);
  setGenerateBusy(true);
  startLoadingSequence(["Opening saved session...", "Restoring structured notes...", "Preparing study workspace..."]);
  showNotesSkeleton("Loading saved notes...");

  try {
    const data = await apiRequest(`/api/history/${id}`);
    initializeResultState(data);
    await renderResult(data);
    setStatus("Loaded from history.");
  } catch (error) {
    setStatus(normalizeErrorMessage(error.message), true);
    if (state.currentResult) {
      await renderResult(state.currentResult);
    } else {
      renderNotesEmptyState();
    }
  } finally {
    setLoading(false);
    setGenerateBusy(false);
    setLanguageEnabled(Boolean(state.sourceResult));
  }
}

function handleFeedbackSubmit(event) {
  event.preventDefault();

  const name = String(els.feedbackName?.value || "").trim();
  const email = String(els.feedbackEmail?.value || "").trim();
  const message = String(els.feedbackMessage?.value || "").trim();

  if (!message) {
    setFeedbackStatus("Please write a message before saving feedback.", "error");
    return;
  }

  setButtonLoading(els.feedbackSubmitBtn, true, "Saving...");

  try {
    const existing = JSON.parse(localStorage.getItem("studykit_feedback_entries") || "[]");
    existing.unshift({
      name,
      email,
      message,
      created_at: new Date().toISOString(),
      student_name: state.currentStudent?.full_name || "",
    });

    localStorage.setItem("studykit_feedback_entries", JSON.stringify(existing.slice(0, 20)));
    persistFeedbackDraft();
    setFeedbackStatus(
      "Feedback saved on this device. The UI is ready to connect to an API or email service later.",
      "success"
    );
  } catch (error) {
    setFeedbackStatus("Unable to save feedback locally in this browser.", "error");
  } finally {
    setButtonLoading(els.feedbackSubmitBtn, false);
  }
}

function attachRipple(event) {
  const target = event.target.closest("button, .toc-link");
  if (!target || target.disabled) return;

  const rect = target.getBoundingClientRect();
  const ripple = document.createElement("span");
  const size = Math.max(rect.width, rect.height);

  ripple.className = "ripple";
  ripple.style.width = `${size}px`;
  ripple.style.height = `${size}px`;
  ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
  ripple.style.top = `${event.clientY - rect.top - size / 2}px`;

  target.appendChild(ripple);
  ripple.addEventListener("animationend", () => ripple.remove(), { once: true });
}

if (els.authTabLogin) {
  els.authTabLogin.addEventListener("click", () => setAuthMode("login"));
}

if (els.authTabRegister) {
  els.authTabRegister.addEventListener("click", () => setAuthMode("register"));
}

if (els.loginForm) {
  els.loginForm.addEventListener("submit", handleLoginSubmit);
}

if (els.registerForm) {
  els.registerForm.addEventListener("submit", handleRegisterSubmit);
}

if (els.logoutBtn) {
  els.logoutBtn.addEventListener("click", () => {
    clearSession();
    setAuthMode("login");
    setAuthStatus("Logged out successfully.");
  });
}

if (els.generateForm) {
  els.generateForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const url = String(els.youtubeUrl?.value || "").trim();
    if (!url) {
      setStatus("Please paste a YouTube URL.", true);
      return;
    }
    generateNotes(url);
  });
}

if (els.heroGenerateBtn) {
  els.heroGenerateBtn.addEventListener("click", () => {
    focusGenerateWorkspace();
  });
}

if (els.emptyGenerateBtn) {
  els.emptyGenerateBtn.addEventListener("click", () => {
    focusGenerateWorkspace();
  });
}

if (els.heroDemoBtn) {
  els.heroDemoBtn.addEventListener("click", () => {
    openDemoModal();
  });
}

if (els.closeDemoBtn) {
  els.closeDemoBtn.addEventListener("click", () => {
    closeDemoModal();
  });
}

if (els.demoModal) {
  els.demoModal.addEventListener("click", (event) => {
    const closeTarget = event.target;
    if (closeTarget instanceof HTMLElement && closeTarget.dataset.closeDemo === "true") {
      closeDemoModal();
    }
  });
}

if (els.languageSelect) {
  els.languageSelect.addEventListener("change", () => {
    switchLanguage(els.languageSelect.value);
  });
}

if (els.downloadPdfBtn) {
  els.downloadPdfBtn.addEventListener("click", () => {
    exportNotesToPdf();
  });
}

if (els.refreshHistoryBtn) {
  els.refreshHistoryBtn.addEventListener("click", () => {
    loadHistory();
  });
}

if (els.historySearch) {
  els.historySearch.addEventListener("input", () => {
    state.historySearch = String(els.historySearch.value || "").trim().toLowerCase();
    applyHistoryFilters();
  });
}

if (els.historyFilter) {
  els.historyFilter.addEventListener("change", () => {
    state.historyFilter = String(els.historyFilter.value || "all");
    applyHistoryFilters();
  });
}

if (els.feedbackForm) {
  els.feedbackForm.addEventListener("submit", handleFeedbackSubmit);
}

[els.feedbackName, els.feedbackEmail, els.feedbackMessage].forEach((input) => {
  if (!input) return;
  input.addEventListener("input", () => {
    persistFeedbackDraft();
  });
});

if (els.themeToggle) {
  els.themeToggle.addEventListener("click", () => {
    const next = document.body.dataset.theme === "dark" ? "light" : "dark";
    setTheme(next);
    if (state.currentResult) {
      renderResult(state.currentResult);
    }
  });
}

document.addEventListener("click", attachRipple);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeDemoModal();
  }
});

setTheme(getSavedTheme());
setAuthMode("login");
applyAuthUi();
restoreFeedbackDraft();
startHeroTyping();
initializeRevealAnimations();
restoreSession();
