'use strict';



/**
 * navbar toggle
 */

const overlay = document.querySelector("[data-overlay]");
const navOpenBtn = document.querySelector("[data-nav-open-btn]");
const navbar = document.querySelector("[data-navbar]");
const navCloseBtn = document.querySelector("[data-nav-close-btn]");

const navElems = [overlay, navOpenBtn, navCloseBtn];

for (let i = 0; i < navElems.length; i++) {
  navElems[i].addEventListener("click", function () {
    navbar.classList.toggle("active");
    overlay.classList.toggle("active");
  });
}



/**
 * header & go top btn active on page scroll
 */

const header = document.querySelector("[data-header]");
const goTopBtn = document.querySelector("[data-go-top]");

window.addEventListener("scroll", function () {
  if (window.scrollY >= 80) {
    header.classList.add("active");
    goTopBtn.classList.add("active");
  } else {
    header.classList.remove("active");
    goTopBtn.classList.remove("active");
  }
});
'use strict';





/**
 * Function to toggle the chatbot display
 */
function toggleChatbot() {
    const chatbotContainer = document.getElementById("chatbotContainer");
    chatbotContainer.style.display = chatbotContainer.style.display === "none" ? "block" : "none";
}

/**
 * Function to redirect to the chat page with username and language
 */
function startChat() {
    const username = document.getElementById("username").value;
    const language = document.getElementById("language").value;

    if (username) {
        window.location.href = `/start-chat?username=${encodeURIComponent(username)}&language=${encodeURIComponent(language)}`;
    } else {
        alert("Please enter your username to start chatting.");
    }
}

/**
 * Load documents for the chatbot context
 */
async function loadDocuments() {
    const formData = new FormData(document.getElementById('fileUploadForm'));
    const response = await fetch('/load_documents', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    document.getElementById('loadOutput').innerText = result.message;
}

/**
 * Display a welcome message from the assistant
 */
function displayWelcomeMessage() {
  const chatBox = document.getElementById('chatbot');
  const welcomeMessageElement = document.createElement('div');
  welcomeMessageElement.classList.add('chat-message', 'assistant-message');
  welcomeMessageElement.innerHTML = `
      <img src="/static/images/v-logo.png" alt="Assistant" class="profile-image">
      <div class="message-bubble">${welcomeMessage}</div>`;  // Use translated message
  chatBox.appendChild(welcomeMessageElement);
}

/**
 * Function to send a question to the server and receive a response
 */
'use strict';

/**
 * Function to send a question to the server and receive a response
 */
async function askQuestion() {
    const question = document.getElementById('questionInput').value;
    const chatBox = document.getElementById('chatbot');

    if (!question.trim()) return; // Prevent empty submissions

    // Display the user's message
    const userMessage = document.createElement('div');
    userMessage.classList.add('chat-message', 'user-message');
    userMessage.innerHTML = `
        <img src="/static/images/user.png" alt="User" class="profile-image">
        <div class="message-bubble">${question}</div>`;
    chatBox.appendChild(userMessage);
    document.getElementById('questionInput').value = ''; // Clear input field

    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom

    // Fetch response from the server, including the language parameter
    const response = await fetch('/ask_question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, language })
    });
    const result = await response.json();

    // Display the assistant's response with a voice button
    const assistantMessage = document.createElement('div');
    assistantMessage.classList.add('chat-message', 'assistant-message');
    assistantMessage.innerHTML = `
        <img src="/static/images/v-logo.png" alt="Assistant" class="profile-image">
        <div class="message-bubble">${result.response}</div>
        <button class="voice-button" onclick="speakText('${result.response}')">
            <i class="fas fa-volume-up"></i>
        </button>`;
    chatBox.appendChild(assistantMessage);

    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
}

/**
 * Function to use Web Speech API to read the text
 */
function speakText(text) {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language === "Spanish" ? "es-ES" :
                     language === "German" ? "de-DE" :
                     language === "Hindi" ? "hi-IN" :
                     language === "French" ? "fr-FR" : 
                     language=== "English" ? "en-US": // Set language based on selection
    synth.speak(utterance);
}

/**
 * Function to clear the chat history
 */
async function clearAll() {
    await fetch('/clear_all', { method: 'POST' });
    document.getElementById('chatbot').innerHTML = '';
    displayWelcomeMessage(); // Re-display welcome message after clearing
}

/**
 * Event listener to handle Enter key press for question input
 */
document.getElementById('questionInput').addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the default form submission
        askQuestion(); // Call askQuestion function
    }
});

/**
 * Function to handle speech recognition for input
 */
function startListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Speech recognition is not supported in this browser.");
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = language || 'en-US'; // Use selected language or default to English
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.start();

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById('questionInput').value = transcript;
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
    };
}
