const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");

function appendMessage(text, sender) {
  const message = document.createElement("div");
  message.className = `message ${sender}`;
  message.innerHTML = `
    <div class="message-bubble">${text}</div>
  `;
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

chatForm.addEventListener("submit", async function (e) {
  e.preventDefault();

  const text = chatInput.value.trim();
  if (!text) return;

  appendMessage(text, "user");
  chatInput.value = "";

  appendMessage("جاري تحليل الطلب...", "bot");

  try {
    const response = await fetch("/advisor", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message: text })
    });

    const data = await response.json();

    chatMessages.removeChild(chatMessages.lastChild);
    appendMessage(data.reply || "لم يتم إرجاع رد.", "bot");
  } catch (error) {
    chatMessages.removeChild(chatMessages.lastChild);
    appendMessage("تعذر الاتصال بالخادم.", "bot");
    console.error(error);
  }
});