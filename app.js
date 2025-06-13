// Emotion emoji mapping
const emotionEmojis = {
    joy: "😂",
    sadness: "😢",
    anger: "😡",
    disgust: "🤢",
    fear: "😱",
    surprise: "😮",
    neutral: "😐"
};

// Reaction emojis for the picker
const reactionEmojis = ['❤️','👎', '👍', '👏','😍', '😮', '😢', '😡'];

// DOM Elements
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const usernameInput = document.getElementById('usernameInput');
const chatContainer = document.getElementById('chatContainer');

// API endpoint
const API_URL = 'http://localhost:8001/predict';
const MESSAGES_URL = 'http://localhost:8001/messages';

// Store chat history for reload
let chatHistory = [];

// Store reactions for each message
let messageReactions = new Map();

// Ekman mapping for tooltip
const ekmanDescriptions = {
    anger: "anger, annoyance, disapproval",
    disgust: "disgust",
    fear: "fear, nervousness",
    joy: "joy, amusement, approval, excitement, gratitude, love, optimism, pride, relief, caring, admiration, desire",
    sadness: "sadness, disappointment, embarrassment, grief, remorse",
    surprise: "surprise, realization, curiosity, confusion",
    neutral: "neutral"
};

// Add a message to the chat
function addMessage(text, isUser = true, emotions = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `bubble ${isUser ? 'bubble-user' : 'bubble-bot'}`;
    messageDiv.dataset.messageId = Date.now().toString();
    
    if (isUser) {
        // Modern animated dots loading
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-dots';
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'loading-dot';
            loadingDiv.appendChild(dot);
        }
        messageDiv.appendChild(loadingDiv);
    } else if (emotions === 'loading') {
        // Modern animated dots loading
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-dots';
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'loading-dot';
            loadingDiv.appendChild(dot);
        }
        messageDiv.appendChild(loadingDiv);
    } else {
        // Simple emotion bars for all emotions (no tooltip)
        const emotionBars = document.createElement('div');
        emotionBars.className = 'emotion-bars';
        // Lọc chỉ các nhãn được bật 1 (probability > threshold)
        const selectedEmotions = emotions.filter(emotion => emotion.probability > emotion.threshold);
        selectedEmotions.forEach((emotion, idx) => {
            const bar = document.createElement('div');
            bar.className = 'emotion-bar';
            bar.setAttribute('data-emo', emotion.emotion);
            // Emoji
            const emoji = document.createElement('span');
            emoji.className = 'emotion-bar-emoji';
            emoji.textContent = emotionEmojis[emotion.emotion] || '';
            bar.appendChild(emoji);
            // Label
            const label = document.createElement('span');
            label.className = 'emotion-bar-label';
            label.textContent = emotion.emotion.charAt(0).toUpperCase() + emotion.emotion.slice(1);
            // Tooltip
            const tooltip = document.createElement('span');
            tooltip.className = 'emotion-tooltip';
            tooltip.textContent = ekmanDescriptions[emotion.emotion] || '';
            label.appendChild(tooltip);
            bar.appendChild(label);
            emotionBars.appendChild(bar);
        });
        messageDiv.appendChild(emotionBars);
        // Insert reaction system after emotion bars
        addReactionSystem(messageDiv);
    }
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    chatContainer.appendChild(errorDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Remove error message after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Show modern animated dots loading
function showLoadingDots() {
    addMessage('', false, 'loading');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Remove last bot message (used for loading)
function removeLastBotMessage() {
    const bubbles = chatContainer.querySelectorAll('.bubble-bot');
    if (bubbles.length > 0) {
        bubbles[bubbles.length - 1].remove();
    }
}

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = usernameInput.value.trim();
    const text = userInput.value.trim();
    if (!username || !text) return;
    
    // Clear input
    userInput.value = '';
    
    // Only push to chatHistory, do NOT call addMessage here
    chatHistory.push({ username, text });
    
    // Show loading dots
    showLoadingDots();
    
    try {
        // Make API request
        await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, text })
        });
        
        // Remove loading dots
        removeLastBotMessage();
        
    } catch (error) {
        removeLastBotMessage();
        console.error('Error:', error);
        showError('Failed to connect to the server. Please try again.');
    }
});

// Reload (regenerate) last bot response
const reloadBtn = document.getElementById('reloadBtn');
if (reloadBtn) {
    reloadBtn.addEventListener('click', async () => {
        if (chatHistory.length === 0) return;
        const last = chatHistory[chatHistory.length - 1];
        // Remove last bot message
        removeLastBotMessage();
        // Show loading dots
        showLoadingDots();
        try {
            await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username: last.username, text: last.text })
            });
            removeLastBotMessage();
        } catch (error) {
            removeLastBotMessage();
            showError('Failed to connect to the server. Please try again.');
        }
    });
}

// New conversation (clear chat)
const newChatBtn = document.getElementById('newChatBtn');
if (newChatBtn) {
    newChatBtn.addEventListener('click', () => {
        chatContainer.innerHTML = '';
        chatHistory = [];
    });
}

// Handle input focus
userInput.addEventListener('focus', () => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
});

// Add reaction functionality to a message
function addReactionSystem(messageDiv) {
    const reactionContainer = document.createElement('div');
    reactionContainer.className = 'reaction-container';
    
    // Add reaction trigger button
    const trigger = document.createElement('button');
    trigger.className = 'reaction-trigger';
    trigger.innerHTML = '➕';
    reactionContainer.appendChild(trigger);
    
    // Create reaction picker
    const picker = document.createElement('div');
    picker.className = 'reaction-picker';
    reactionEmojis.forEach(emoji => {
        const emojiBtn = document.createElement('span');
        emojiBtn.className = 'reaction-emoji';
        emojiBtn.textContent = emoji;
        emojiBtn.onclick = (e) => {
            e.stopPropagation();
            addReaction(messageDiv, emoji);
            picker.classList.remove('active');
        };
        picker.appendChild(emojiBtn);
    });
    reactionContainer.appendChild(picker);
    
    // Create reactions display
    const reactionsDisplay = document.createElement('div');
    reactionsDisplay.className = 'reactions-display';
    reactionContainer.appendChild(reactionsDisplay);
    
    // Add click handler for the trigger
    trigger.onclick = (e) => {
        e.stopPropagation();
        picker.classList.toggle('active');
    };
    
    // Close picker when clicking outside
    document.addEventListener('click', (e) => {
        if (!reactionContainer.contains(e.target)) {
            picker.classList.remove('active');
        }
    });
    
    messageDiv.appendChild(reactionContainer);
    
    // Initialize reactions for this message if they exist
    const messageId = messageDiv.dataset.messageId || Date.now().toString();
    messageDiv.dataset.messageId = messageId;
    if (messageReactions.has(messageId)) {
        updateReactionsDisplay(reactionsDisplay, messageReactions.get(messageId));
    } else {
        messageReactions.set(messageId, new Map());
    }
}

// Add a reaction to a message
function addReaction(messageDiv, emoji) {
    const messageId = messageDiv.dataset.messageId;
    const reactions = messageReactions.get(messageId);
    const count = reactions.get(emoji) || 0;
    reactions.set(emoji, count + 1);
    
    const reactionsDisplay = messageDiv.querySelector('.reactions-display');
    updateReactionsDisplay(reactionsDisplay, reactions);
}

// Update the reactions display
function updateReactionsDisplay(container, reactions) {
    container.innerHTML = '';
    reactions.forEach((count, emoji) => {
        if (count > 0) {
            const badge = document.createElement('div');
            badge.className = 'reaction-badge';
            badge.innerHTML = `
                <span class="emoji">${emoji}</span>
                <span class="count">${count}</span>
            `;
            container.appendChild(badge);
        }
    });
}

// Restore color mode toggle logic
const colorModeBtn = document.getElementById('colorModeBtn');
const colorModeIcon = document.getElementById('colorModeIcon');
if (colorModeBtn) {
    colorModeBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        // Toggle icon (sun/moon)
        if (document.body.classList.contains('dark-mode')) {
            colorModeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 1 0 9.79 9.79z"/>';
        } else {
            colorModeIcon.innerHTML = '<circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>';
        }
    });
}

const TOP_EMOTIONS = 7;

let lastMessageId = null;

function addMessageToChat(message) {
    // Emma luôn bên trái, Lucas luôn bên phải
    const isLucas = message.username === "Lucas";
    const bubble = document.createElement('div');
    bubble.className = isLucas ? 'bubble bubble-user' : 'bubble bubble-other';

    // Username
    const nameDiv = document.createElement('span');
    nameDiv.className = 'username';
    nameDiv.textContent = message.username;
    bubble.appendChild(nameDiv);

    // Message text
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = message.text;
    bubble.appendChild(textDiv);

    // Emotion bars (top N)
    const emotionBars = document.createElement('div');
    emotionBars.className = 'emotion-bars';
    // Lọc chỉ các nhãn được bật 1 (probability > threshold)
    const selectedEmotions = message.emotions.filter(e => e.probability > e.threshold);
    selectedEmotions.forEach((emotion) => {
        const bar = document.createElement('div');
        bar.className = 'emotion-bar';
        bar.setAttribute('data-emo', emotion.emotion);
        // Emoji
        const emoji = document.createElement('span');
        emoji.className = 'emotion-bar-emoji';
        emoji.textContent = emotionEmojis[emotion.emotion] || '';
        bar.appendChild(emoji);
        // Label
        const label = document.createElement('span');
        label.className = 'emotion-bar-label';
        label.textContent = emotion.emotion.charAt(0).toUpperCase() + emotion.emotion.slice(1);
        // Tooltip
        const tooltip = document.createElement('span');
        tooltip.className = 'emotion-tooltip';
        tooltip.textContent = ekmanDescriptions[emotion.emotion] || '';
        label.appendChild(tooltip);
        bar.appendChild(label);
        emotionBars.appendChild(bar);
    });
    bubble.appendChild(emotionBars);

    addReactionSystem(bubble);

    chatContainer.appendChild(bubble);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

let lastMessageCount = 0;

function appendNewMessages(messages) {
    for (let i = lastMessageCount; i < messages.length; i++) {
        addMessageToChat(messages[i]);
    }
    lastMessageCount = messages.length;
}

async function pollMessages() {
    try {
        const response = await fetch(MESSAGES_URL);
        if (!response.ok) throw new Error('Failed to fetch messages');
        const messages = await response.json();
        appendNewMessages(messages);
    } catch (err) {
        // Xử lý lỗi nếu cần
    }
    setTimeout(pollMessages, 1500);
}
pollMessages(); 