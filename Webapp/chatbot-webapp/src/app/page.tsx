"use client";

import { useState } from "react";
import ModelSelector from "../components/ModelSelector";

export default function Home() {
  const [inputMessage, setInputMessage] = useState("");
  const [chatMessages, setChatMessages] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("Model1");

  const handleSendMessage = async () => {
    if (inputMessage.trim()) {
      const userMessage = `You: ${inputMessage}`;
      setChatMessages((prevMessages) => [...prevMessages, userMessage]);
      setInputMessage("");
      setIsLoading(true);

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: inputMessage, model: selectedModel }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setChatMessages((prevMessages) => [...prevMessages, `The predicted class for "${data.requirement}" is "${data.predicted_class}"`]);
      } catch (error) {
        console.error("Error sending message:", error);
        setChatMessages((prevMessages) => [
          ...prevMessages,
          "Bot: Sorry, something went wrong.",
        ]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      <header className="bg-blue-600 text-white p-4 shadow-md flex justify-between items-center">
        <h1 className="text-xl font-semibold">Chatbot</h1>
        <ModelSelector
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
        />
      </header>
      <main className="flex-1 overflow-y-auto p-4 space-y-2">
        {chatMessages.map((message, index) => (
          <div
            key={index}
            className={`p-2 rounded-lg max-w-xs ${
              message.startsWith("You:")
                ? "bg-blue-500 text-white self-end ml-auto"
                : "bg-gray-300 text-black dark:bg-gray-700 dark:text-white self-start mr-auto"
            }`}
          >
            {message}
          </div>
        ))}
        {isLoading && (
          <div className="p-2 rounded-lg bg-gray-300 text-black dark:bg-gray-700 dark:text-white self-start mr-auto">
            Bot: Thinking...
          </div>
        )}
      </main>
      <footer className="bg-white dark:bg-gray-800 p-4 border-t border-gray-200 dark:border-gray-700 flex items-center">
        <input
          type="text"
          className="flex-1 p-2 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          placeholder="Type your message..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === "Enter" && !isLoading) {
              handleSendMessage();
            }
          }}
          disabled={isLoading}
        />
        <button
          className="bg-blue-600 text-white p-2 rounded-r-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          onClick={handleSendMessage}
          disabled={isLoading}
        >
          Send
        </button>
      </footer>
    </div>
  );
}


