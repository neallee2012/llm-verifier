const state = {
  threads: [],
  activeThread: null,
  messages: [],
  config: null,
};

const threadList = document.getElementById("thread-list");
const newThreadBtn = document.getElementById("new-thread");
const threadTitleInput = document.getElementById("thread-title");
const saveTitleBtn = document.getElementById("save-title");
const deleteThreadBtn = document.getElementById("delete-thread");
const messagesContainer = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const systemPromptInput = document.getElementById("system-prompt");
const saveSystemPromptBtn = document.getElementById("save-system-prompt");
const primaryModelSelect = document.getElementById("primary-model");
const verifierModelSelect = document.getElementById("verifier-model");
const verifierEnabledToggle = document.getElementById("verifier-enabled");
const verifierToggleStatus = document.getElementById("verifier-toggle-status");
const routingThresholdInput = document.getElementById("routing-threshold");
const routingShortcutToggle = document.getElementById("routing-shortcut-enabled");
const webSearchToggle = document.getElementById("web-search-enabled");
const modelConfigsContainer = document.getElementById("model-configs");
const saveConfigBtn = document.getElementById("save-config");
const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");
let currentStreamController = null;
let isStreaming = false;
let pendingImages = [];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || "Request failed");
  }
  if (response.status === 204) return null;
  return response.json();
}

function escapeHtml(value) {
  return (value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function sanitizeUrl(url) {
  const trimmed = url.trim();
  if (/^(https?:|mailto:)/i.test(trimmed)) return trimmed;
  if (trimmed.startsWith("/")) return trimmed;
  return "#";
}

function renderInlineMarkdown(text) {
  const codeSpans = [];
  let output = text.replace(/`([^`]+)`/g, (_match, code) => {
    const token = `@@CODE${codeSpans.length}@@`;
    codeSpans.push(code);
    return token;
  });

  output = output.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  output = output.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  output = output.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  output = output.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, label, url) => {
    const safeUrl = sanitizeUrl(url);
    return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });
  output = output.replace(/@@CODE(\d+)@@/g, (_match, index) => {
    const code = codeSpans[Number(index)];
    return `<code>${code}</code>`;
  });
  return output;
}

function renderBlocks(text) {
  const lines = text.split(/\r?\n/);
  let html = "";
  let outerListType = null;
  let innerListType = null;
  let inListItem = false;
  let inBlockquote = false;
  let tableLines = [];

  function closeInnerList() {
    if (innerListType) {
      html += `</${innerListType}>`;
      innerListType = null;
    }
  }

  function closeOuterListItem() {
    closeInnerList();
    if (inListItem) {
      html += "</li>";
      inListItem = false;
    }
  }

  function closeList() {
    closeOuterListItem();
    if (outerListType) {
      html += `</${outerListType}>`;
      outerListType = null;
    }
  }

  function closeBlockquote() {
    if (inBlockquote) {
      html += "</blockquote>";
      inBlockquote = false;
    }
  }

  function parseCells(row) {
    return row.replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
  }

  function closeTable() {
    if (tableLines.length === 0) return;
    html += '<div class="table-wrap"><table>';
    const hasSep = tableLines.length >= 2 && /^\|[\s\-:|]+\|$/.test(tableLines[1]);
    if (hasSep) {
      const hCells = parseCells(tableLines[0]);
      html += "<thead><tr>";
      hCells.forEach(c => { html += `<th>${renderInlineMarkdown(c)}</th>`; });
      html += "</tr></thead><tbody>";
      for (let i = 2; i < tableLines.length; i++) {
        const cells = parseCells(tableLines[i]);
        html += "<tr>";
        cells.forEach(c => { html += `<td>${renderInlineMarkdown(c)}</td>`; });
        html += "</tr>";
      }
      html += "</tbody>";
    } else {
      html += "<tbody>";
      tableLines.forEach(row => {
        const cells = parseCells(row);
        html += "<tr>";
        cells.forEach(c => { html += `<td>${renderInlineMarkdown(c)}</td>`; });
        html += "</tr>";
      });
      html += "</tbody>";
    }
    html += "</table></div>";
    tableLines = [];
  }

  lines.forEach((line) => {
    const trimmed = line.trim();
    const indent = line.search(/\S|$/);
    if (!trimmed) {
      closeList();
      closeBlockquote();
      closeTable();
      return;
    }

    if (/^\|.*\|$/.test(trimmed)) {
      closeList();
      closeBlockquote();
      tableLines.push(trimmed);
      return;
    }

    if (tableLines.length > 0) {
      closeTable();
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      closeList();
      closeBlockquote();
      html += "<hr>";
      return;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      closeList();
      closeBlockquote();
      const level = headingMatch[1].length;
      html += `<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`;
      return;
    }

    const quoteMatch = trimmed.match(/^>\s?(.*)$/);
    if (quoteMatch) {
      closeList();
      if (!inBlockquote) {
        html += "<blockquote>";
        inBlockquote = true;
      }
      html += `<p>${renderInlineMarkdown(quoteMatch[1])}</p>`;
      return;
    }

    if (inBlockquote) {
      closeBlockquote();
    }

    const orderedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
    if (orderedMatch) {
      const num = parseInt(orderedMatch[1], 10);
      if (indent >= 2 && outerListType) {
        if (innerListType !== "ol") {
          closeInnerList();
          innerListType = "ol";
          html += `<ol start="${num}">`;
        }
        html += `<li>${renderInlineMarkdown(orderedMatch[2])}</li>`;
      } else {
        if (outerListType !== "ol") {
          closeList();
          outerListType = "ol";
          html += `<ol start="${num}">`;
        } else {
          closeOuterListItem();
        }
        html += `<li>${renderInlineMarkdown(orderedMatch[2])}`;
        inListItem = true;
      }
      return;
    }

    const unorderedMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (unorderedMatch) {
      if (indent >= 2 && outerListType) {
        if (innerListType !== "ul") {
          closeInnerList();
          innerListType = "ul";
          html += "<ul>";
        }
        html += `<li>${renderInlineMarkdown(unorderedMatch[1])}</li>`;
      } else {
        if (outerListType !== "ul") {
          closeList();
          outerListType = "ul";
          html += "<ul>";
        } else {
          closeOuterListItem();
        }
        html += `<li>${renderInlineMarkdown(unorderedMatch[1])}`;
        inListItem = true;
      }
      return;
    }

    closeList();
    closeBlockquote();
    html += `<p>${renderInlineMarkdown(trimmed)}</p>`;
  });

  closeList();
  if (inBlockquote) {
    html += "</blockquote>";
  }
  closeTable();
  return html;
}

function renderMarkdown(text) {
  const escaped = escapeHtml(text);
  const segments = escaped.split("```");
  let html = "";
  segments.forEach((segment, index) => {
    if (index % 2 === 1) {
      const lines = segment.split(/\r?\n/);
      let language = "";
      if (lines.length && /^[\w-]+$/.test(lines[0].trim())) {
        language = lines.shift().trim();
      }
      const code = lines.join("\n");
      html += `<pre><code${language ? ` class=\"language-${language}\"` : ""}>${code}</code></pre>`;
    } else {
      html += renderBlocks(segment);
    }
  });
  return html;
}

function renderThreads() {
  threadList.innerHTML = "";
  state.threads.forEach((thread) => {
    const li = document.createElement("li");
    li.textContent = thread.title || "New thread";
    li.className = thread.id === state.activeThread ? "active" : "";
    li.onclick = () => selectThread(thread.id);
    threadList.appendChild(li);
  });
}

function renderMessages(shouldScroll = true) {
  messagesContainer.innerHTML = "";
  state.messages.forEach((msg) => {
    const div = document.createElement("div");
    div.className = "message";
    if (msg.role === "user") {
      div.classList.add("role-user");
    } else if (msg.role === "assistant") {
      div.classList.add("role-assistant");
    } else if (msg.role === "assistant_draft") {
      div.classList.add("role-draft");
    } else if (msg.role === "verifier") {
      div.classList.add("role-verifier");
    }
    const meta = document.createElement("div");
    meta.className = "meta";
    const status = msg.status ? ` • ${msg.status}` : "";
    meta.textContent = `${msg.role}${msg.model ? ` • ${msg.model}` : ""}${status}`;
    div.appendChild(meta);
    if (msg.images && msg.images.length > 0) {
      const imagesDiv = document.createElement("div");
      imagesDiv.className = "msg-images";
      msg.images.forEach((src) => {
        const img = document.createElement("img");
        img.src = src;
        img.onclick = () => window.open(src, "_blank");
        imagesDiv.appendChild(img);
      });
      div.appendChild(imagesDiv);
    }
    const content = document.createElement("div");
    content.innerHTML = renderMarkdown(msg.content);
    div.appendChild(content);
    messagesContainer.appendChild(div);
  });
  if (shouldScroll) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

function renderConfig() {
  if (!state.config) return;
  if (!state.config.agents) {
    state.config.agents = {
      responder_model_id: state.config.primary_model_id,
      verifier_model_id: state.config.verifier_model_id,
      polisher_model_id: state.config.primary_model_id,
    };
  }
  if (!state.config.routing) {
    state.config.routing = { confidence_threshold: 0.95, enable_verifier_shortcut: true };
  }
  if (!state.config.tools) {
    state.config.tools = { web_search_enabled: true };
  }
  primaryModelSelect.innerHTML = "";
  verifierModelSelect.innerHTML = "";
  modelConfigsContainer.innerHTML = "";

  state.config.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label || model.id;
    primaryModelSelect.appendChild(option.cloneNode(true));
    verifierModelSelect.appendChild(option);

    const block = document.createElement("div");
    block.className = "section";
    const title = document.createElement("h3");
    title.textContent = model.label || model.id;
    block.appendChild(title);

    Object.entries(model).forEach(([key, value]) => {
      if (key === "id" || key === "type" || key === "label") return;
      const field = document.createElement("div");
      field.className = "field";
      const label = document.createElement("label");
      label.textContent = key;
      let input;
      if (key === "instructions") {
        input = document.createElement("textarea");
        input.rows = 3;
      } else {
        input = document.createElement("input");
      }
      input.value = value || "";
      input.dataset.modelId = model.id;
      input.dataset.key = key;
      field.appendChild(label);
      field.appendChild(input);
      block.appendChild(field);
    });

    modelConfigsContainer.appendChild(block);
  });

  primaryModelSelect.value = state.config.primary_model_id;
  verifierModelSelect.value = state.config.verifier_model_id;
  verifierEnabledToggle.checked = !!state.config.verifier_enabled;
  routingThresholdInput.value = state.config.routing.confidence_threshold ?? 0.95;
  routingShortcutToggle.checked = !!state.config.routing.enable_verifier_shortcut;
  webSearchToggle.checked = !!state.config.tools.web_search_enabled;
  updateVerifierToggleUI();
}

function updateVerifierToggleUI() {
  if (!verifierToggleStatus) return;
  const enabled = !!(state.config && state.config.verifier_enabled);
  verifierToggleStatus.textContent = enabled ? "ON" : "OFF";
  verifierToggleStatus.classList.toggle("on", enabled);
  verifierToggleStatus.classList.toggle("off", !enabled);
}

async function loadThreads() {
  state.threads = await api("/api/threads");
  renderThreads();
  if (!state.activeThread && state.threads.length) {
    selectThread(state.threads[0].id);
  }
}

async function selectThread(threadId) {
  state.activeThread = threadId;
  const thread = state.threads.find((t) => t.id === threadId);
  threadTitleInput.value = thread ? thread.title : "";
  renderThreads();
  await loadMessages();
}

async function loadMessages() {
  if (!state.activeThread) return;
  state.messages = await api(`/api/threads/${state.activeThread}/messages`);
  renderMessages();
}

async function createThread() {
  const thread = await api("/api/threads", {
    method: "POST",
    body: JSON.stringify({ title: "New thread" }),
  });
  state.threads.unshift(thread);
  await selectThread(thread.id);
}

async function saveTitle() {
  if (!state.activeThread) return;
  const title = threadTitleInput.value.trim() || "New thread";
  const updated = await api(`/api/threads/${state.activeThread}`, {
    method: "PUT",
    body: JSON.stringify({ title }),
  });
  const idx = state.threads.findIndex((t) => t.id === state.activeThread);
  if (idx >= 0) state.threads[idx] = updated;
  renderThreads();
}

async function deleteThread() {
  if (!state.activeThread) return;
  if (currentStreamController) {
    currentStreamController.abort();
    currentStreamController = null;
  }
  await api(`/api/threads/${state.activeThread}`, { method: "DELETE" });
  state.threads = state.threads.filter((t) => t.id !== state.activeThread);
  state.activeThread = null;
  state.messages = [];
  renderThreads();
  renderMessages();
}

async function sendMessage(event) {
  event.preventDefault();
  if (!state.activeThread) return;
  const content = chatInput.value.trim();
  if (!content && !pendingImages.length) return;
  chatInput.value = "";
  const images = pendingImages.slice();
  pendingImages = [];
  imagePreview.innerHTML = "";
  const userMessage = {
    role: "user",
    content: content || "(image)",
    images,
    model: null,
  };
  state.messages.push(userMessage);
  renderMessages();

  const responderDraftMessage = {
    role: "assistant_draft",
    content: "",
    model: state.config && state.config.agents ? state.config.agents.responder_model_id : null,
  };
  const assistantMessage = {
    role: "assistant",
    content: "",
    model: state.config && state.config.agents ? state.config.agents.polisher_model_id : null,
  };
  let verifierMessage = null;
  state.messages.push(responderDraftMessage);
  state.messages.push(assistantMessage);
  renderMessages();

  if (currentStreamController) {
    currentStreamController.abort();
  }
  currentStreamController = new AbortController();
  const body = {
    content: content || "(image)",
    verifier_enabled: !!(state.config && state.config.verifier_enabled),
  };
  if (images.length) {
    body.images = images;
  }
  const response = await fetch(
    `/api/threads/${state.activeThread}/messages/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: currentStreamController.signal,
    }
  );
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || "Request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  isStreaming = true;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      parts.forEach((chunk) => {
        verifierMessage = handleSseChunk(
          chunk,
          responderDraftMessage,
          assistantMessage,
          verifierMessage
        );
      });
    }
  } finally {
    isStreaming = false;
    currentStreamController = null;
  }
}

function handleSseChunk(chunk, responderDraftMessage, assistantMessage, verifierMessage) {
  const lines = chunk.split("\n");
  let event = "message";
  let data = "";
  lines.forEach((line) => {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data += line.slice(5).trim();
    }
  });
  if (!data) return verifierMessage;
  const payload = JSON.parse(data);
  if (event === "status") {
    if (payload.stage === "responder") {
      responderDraftMessage.status = payload.status;
    } else if (payload.stage === "polisher") {
      assistantMessage.status = payload.status;
    } else if (payload.stage === "verifier") {
      if (!verifierMessage) {
        verifierMessage = {
          role: "verifier",
          content: "",
          model: state.config && state.config.agents ? state.config.agents.verifier_model_id : null,
        };
        const assistantIndex = state.messages.indexOf(assistantMessage);
        if (assistantIndex >= 0) {
          state.messages.splice(assistantIndex, 0, verifierMessage);
        } else {
          state.messages.push(verifierMessage);
        }
      }
      verifierMessage.status = payload.status;
    }
    renderMessages(!isStreaming);
    return verifierMessage;
  }
  if (event === "token") {
    if (payload.stage === "responder") {
      responderDraftMessage.content += payload.delta;
    } else if (payload.stage === "polisher") {
      assistantMessage.content += payload.delta;
    } else if (payload.stage === "verifier" && verifierMessage) {
      verifierMessage.content += payload.delta;
    }
    renderMessages(!isStreaming);
    return verifierMessage;
  }
  if (event === "saved") {
    const msg = payload.message;
    if (msg.role === "assistant") {
      Object.assign(assistantMessage, msg);
    } else if (msg.role === "assistant_draft") {
      Object.assign(responderDraftMessage, msg);
    } else if (msg.role === "verifier" && verifierMessage) {
      Object.assign(verifierMessage, msg);
    }
    renderMessages(!isStreaming);
    return verifierMessage;
  }
  if (event === "routing") {
    return verifierMessage;
  }
  if (event === "title") {
    const updated = payload.thread;
    const idx = state.threads.findIndex((t) => t.id === updated.id);
    if (idx >= 0) {
      state.threads[idx] = updated;
    } else {
      state.threads.unshift(updated);
    }
    if (state.activeThread === updated.id) {
      threadTitleInput.value = updated.title || "";
    }
    renderThreads();
    return verifierMessage;
  }
  if (event === "done") {
    isStreaming = false;
    return verifierMessage;
  }
  if (event === "error") {
    alert(payload.message || "Streaming error");
  }
  return verifierMessage;
}

async function loadSystemPrompt() {
  const data = await api("/api/system-prompt");
  systemPromptInput.value = data.content || "";
}

async function saveSystemPrompt() {
  const content = systemPromptInput.value;
  await api("/api/system-prompt", {
    method: "PUT",
    body: JSON.stringify({ content }),
  });
}

async function loadConfig() {
  state.config = await api("/api/config");
  renderConfig();
}

async function saveConfig() {
  const updated = JSON.parse(JSON.stringify(state.config));
  updated.primary_model_id = primaryModelSelect.value;
  updated.verifier_model_id = verifierModelSelect.value;
  updated.verifier_enabled = verifierEnabledToggle.checked;
  updated.agents = updated.agents || {};
  updated.agents.responder_model_id = primaryModelSelect.value;
  updated.agents.verifier_model_id = verifierModelSelect.value;
  if (!updated.agents.polisher_model_id) {
    updated.agents.polisher_model_id = primaryModelSelect.value;
  }
  updated.routing = updated.routing || {};
  updated.routing.confidence_threshold = Number(routingThresholdInput.value || "0.95");
  updated.routing.enable_verifier_shortcut = !!routingShortcutToggle.checked;
  updated.tools = updated.tools || {};
  updated.tools.web_search_enabled = !!webSearchToggle.checked;
  document
    .querySelectorAll("#model-configs input, #model-configs textarea")
    .forEach((input) => {
    const model = updated.models.find((m) => m.id === input.dataset.modelId);
    if (model) {
      model[input.dataset.key] = input.value;
    }
    });
  state.config = await api("/api/config", {
    method: "PUT",
    body: JSON.stringify(updated),
  });
  renderConfig();
}

newThreadBtn.addEventListener("click", createThread);
saveTitleBtn.addEventListener("click", saveTitle);
deleteThreadBtn.addEventListener("click", deleteThread);
chatForm.addEventListener("submit", sendMessage);
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});
saveSystemPromptBtn.addEventListener("click", saveSystemPrompt);
saveConfigBtn.addEventListener("click", saveConfig);
verifierEnabledToggle.addEventListener("change", () => {
  if (!state.config) return;
  state.config.verifier_enabled = verifierEnabledToggle.checked;
  updateVerifierToggleUI();
});

imageInput.addEventListener("change", () => {
  const files = Array.from(imageInput.files);
  imageInput.value = "";
  files.forEach((file) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => {
      pendingImages.push(reader.result);
      renderImagePreview();
    };
    reader.readAsDataURL(file);
  });
});

function renderImagePreview() {
  imagePreview.innerHTML = "";
  pendingImages.forEach((src, idx) => {
    const item = document.createElement("div");
    item.className = "preview-item";
    const img = document.createElement("img");
    img.src = src;
    const btn = document.createElement("button");
    btn.className = "remove-btn";
    btn.textContent = "×";
    btn.onclick = () => {
      pendingImages.splice(idx, 1);
      renderImagePreview();
    };
    item.appendChild(img);
    item.appendChild(btn);
    imagePreview.appendChild(item);
  });
}

Promise.all([loadThreads(), loadSystemPrompt(), loadConfig()]).catch((error) => {
  console.error(error);
  alert(error.message);
});
