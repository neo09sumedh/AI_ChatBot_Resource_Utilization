import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

const lexApiUrl = "https://98bj0c1be6.execute-api.us-east-1.amazonaws.com/Development/lex"; // Replace with your Lex API Gateway endpoint
const dynamoApiUrl = "https://0r398xc5x9.execute-api.us-east-1.amazonaws.com/Development/DynamodbManger"; // Replace with your DynamoDB check API Gateway endpoint

function App() {
  const [userMessage, setUserMessage] = useState("");
  const [messages, setMessages] = useState([{ text: "Hi! I'm your Lex assistant. How can I help?", sender: "bot" }]);
  const [sessionId, setSessionId] = useState(`session-${Date.now()}`);
  const [isWaiting, setIsWaiting] = useState(false); // Flag to indicate waiting for response

  const appendMessage = (text, sender = 'bot') => {
    setMessages((prevMessages) => [...prevMessages, { text, sender }]);
  };

  const sendMessage = async () => {
    if (!userMessage.trim()) return;

    // Show user message
    appendMessage(userMessage, "user");
    setUserMessage("");

    // Show waiting message
    setIsWaiting(true);
    appendMessage("Waiting for response...", "bot");

    // Step 1: Send message to Lex (via Lex API Gateway)
    try {
      const lexResponse = await axios.post(lexApiUrl, {
        message: userMessage,
        sessionId: sessionId // sessionId used to track conversation
      });

      // Assuming Lex response is immediately returned with confirmation (or no response, since it goes to SQS)
      appendMessage("Your message is being processed, please wait...", "bot");

      // Step 2: Polling for DynamoDB response (via DynamoDB API Gateway)
      
      const pollForResponse = async () => {
        try {
          const response = await axios.post(dynamoApiUrl, {
            sessionId: sessionId
          });
          // Parse the response body (it's a stringified JSON)
          const parsedBody = typeof response.data.body === 'string'
            ? JSON.parse(response.data.body)
            : response.data.body;
          const responseItem = parsedBody.responses?.[0];
          
          if (responseItem?.status  === 'ready') {
            setIsWaiting(false);
            appendMessage(`Processed: ${responseItem.response}`, "bot"); // ✅ This is now a string
          } else {
            setTimeout(pollForResponse, 30); // keep polling
          }
        } catch (error) {
          console.error("Error checking DynamoDB response:", error);
          setIsWaiting(false);
          appendMessage("Sorry, there was an error fetching data. Please try again.", "bot");
        }
      };

      // Start polling right after sending the message to Lex
      setTimeout(pollForResponse, 3000);

    } catch (error) {
      appendMessage(error,"bot");
      console.error("Error sending message to Lex:", error);
      setIsWaiting(false);
      appendMessage("Sorry, there was an error. Please try again.", "bot");
      console.error("Error sending message to Lex:", error);
  // Check if error.response exists and print response data
      if (error.response) {
        console.error("Error response data:", error.response.data);
        console.error("Error status:", error.response.status);
      }
      // Check if error.request exists
      else if (error.request) {
        console.error("No response received:", error.request);
      }
      // If it's an unknown error
      else {
        console.error("Error message:", error.message);
     }

       setIsWaiting(false);
        appendMessage("Sorry, there was an error. Please try again.", "bot");
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        <div className="messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message ${msg.sender === 'user' ? 'user' : 'bot'}`}
            >
              {msg.text}
            </div>
          ))}
        </div>

        <div className="input-area">
          <input
            type="text"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            placeholder="Type a message..."
          />
          <button onClick={sendMessage}>Send</button>
        </div>

        {isWaiting && (
          <div className="spinner-container">
            <div className="spinner"></div>
            <p>Waiting for response...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
