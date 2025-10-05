async function sendMessage() {
const input = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const message = input.value.trim();
if (!message) return;

chatBox.innerHTML += `<div class='user'>Tú: ${message}</div>`;
input.value = "";

const res = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ message })
});

const data = await res.json();
chatBox.innerHTML += `<div class='bot'>UCVia: ${data.response}</div>`;
chatBox.scrollTop = chatBox.scrollHeight;
}
