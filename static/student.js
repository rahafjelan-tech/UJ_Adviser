const messagesWrap = document.getElementById("messagesWrap");
const chatInput    = document.getElementById("chatInput");
const sendBtn      = document.getElementById("sendBtn");
const welcomeCard  = document.getElementById("welcomeCard");

let isWaiting = false;

// ── Auto-resize textarea ──
chatInput.addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 120) + "px";
});

// ── Send on Enter ──
chatInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function sendSuggestion(btn) {
  chatInput.value = btn.textContent.trim();
  sendMessage();
}

function clearChat() {
  const wc = document.getElementById("welcomeCard");
  messagesWrap.innerHTML = "";
  if (wc) {
    wc.style.display = "";
    messagesWrap.appendChild(wc);
  }
}

function appendMessage(text, role) {
  const wc = document.getElementById("welcomeCard");
  if (wc) wc.style.display = "none";

  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar-sm";
  avatar.textContent = role === "user" ? "أ" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = formatText(text);

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesWrap.appendChild(row);
  scrollToBottom();
  return row;
}

function showTyping(msg) {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.id = "typingRow";

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar-sm";
  avatar.textContent = "🤖";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  if (msg) {
    bubble.innerHTML = formatText(msg);
  } else {
    bubble.innerHTML = `
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>`;
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesWrap.appendChild(row);
  scrollToBottom();
}

function removeTyping() {
  const el = document.getElementById("typingRow");
  if (el) el.remove();
}

function updateTyping(msg) {
  const el = document.getElementById("typingRow");
  if (el) {
    const bubble = el.querySelector(".msg-bubble");
    if (bubble) bubble.innerHTML = formatText(msg);
  }
}

function formatText(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

function scrollToBottom() {
  messagesWrap.scrollTop = messagesWrap.scrollHeight;
}

// ── Check if system is ready, retry if loading ──
async function waitForReady(maxWaitMs = 180000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    try {
      const res  = await fetch("/health");
      const data = await res.json();
      if (data.status === "ready") return true;
      if (data.status === "error") return false;
    } catch (e) { /* ignore */ }
    await new Promise(r => setTimeout(r, 4000));
  }
  return false;
}

async function sendMessage() {
  if (isWaiting) return;

  const text = chatInput.value.trim();
  if (!text) return;

  isWaiting = true;
  sendBtn.disabled = true;
  chatInput.value = "";
  chatInput.style.height = "auto";

  appendMessage(text, "user");

  // Check system status first
  try {
    const healthRes  = await fetch("/health");
    const healthData = await healthRes.json();

    if (healthData.status === "loading") {
      showTyping("⏳ النظام يتهيأ حالياً، انتظر قليلاً...");
      const ready = await waitForReady();
      if (!ready) {
        updateTyping("⚠️ استغرق التحميل وقتاً طويلاً. يرجى إعادة المحاولة.");
        isWaiting = false;
        sendBtn.disabled = false;
        return;
      }
      updateTyping("✅ النظام جاهز! جاري معالجة سؤالك...");
    } else {
      showTyping();
    }
  } catch (e) {
    showTyping();
  }

  try {
    const res  = await fetch("/student", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text })
    });
    const data = await res.json();
    removeTyping();
    appendMessage(data.reply || "لم يتم إرجاع رد.", "bot");
  } catch (err) {
    removeTyping();
    appendMessage("⚠️ تعذّر الاتصال بالخادم.", "bot");
    console.error(err);
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}
