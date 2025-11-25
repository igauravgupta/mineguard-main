import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import ChatHistorySidebar from "../../components/ChatAI/ChatHistorySidebar";
import MessageList from "../../components/ChatAI/MessageList";
import ChatInput from "../../components/ChatAI/ChatInput";
import ChatHeader from "../../components/ChatAI/ChatHeader";

const ChatbotPage = () => {
  const [chatHistory, setChatHistory] = useState(() => {
    const saved = localStorage.getItem("chatHistory");
    return saved ? JSON.parse(saved) : [];
  });
  const [currentChatId, setCurrentChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [chatToDelete, setChatToDelete] = useState(null);
  const messagesEndRef = useRef(null);
  const _navigate = useNavigate();

  useEffect(() => {
    if (chatHistory.length === 0) {
      createNewChat();
    } else {
      loadChat(chatHistory[0].id);
    }
  }, []);

  useEffect(() => {
    if (messages.length > 0 && currentChatId) {
      localStorage.setItem(
        `chatMessages_${currentChatId}`,
        JSON.stringify(messages)
      );
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, currentChatId]);

  const createNewChat = () => {
    const newChatId = Date.now().toString();
    const newChat = {
      id: newChatId,
      title: "New Chat",
      timestamp: new Date().toISOString(),
    };

    setChatHistory((prev) => [newChat, ...prev]);
    setCurrentChatId(newChatId);
    setMessages([]);
    setShowHistory(false);
    saveChatHistory([newChat, ...chatHistory]);
  };

  const loadChat = (chatId) => {
    const savedMessages = localStorage.getItem(`chatMessages_${chatId}`);
    setCurrentChatId(chatId);
    setMessages(savedMessages ? JSON.parse(savedMessages) : []);
    setShowHistory(false);
  };

  const deleteChat = (chatId) => {
    if (window.confirm("Are you sure you want to delete this chat?")) {
      const updatedHistory = chatHistory.filter((chat) => chat.id !== chatId);
      setChatHistory(updatedHistory);
      saveChatHistory(updatedHistory);

      localStorage.removeItem(`chatMessages_${chatId}`);

      if (chatId === currentChatId) {
        if (updatedHistory.length > 0) {
          loadChat(updatedHistory[0].id);
        } else {
          createNewChat();
        }
      }
    }
    setChatToDelete(null);
  };

  const saveChatHistory = (history) => {
    localStorage.setItem("chatHistory", JSON.stringify(history));
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { text: input, isUser: true };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(
        "https://ptzzhq85-8000.inc1.devtunnels.ms/search",
        { query: input },
        { headers: { "Content-Type": "application/json" } }
      );

      let aiResponse = "I didn't understand that";
      if (
        Array.isArray(response.data?.results) &&
        response.data.results.length > 0
      ) {
        const best = response.data.results.reduce(
          (max, curr) => (curr.score > max.score ? curr : max),
          response.data.results[0]
        );
        aiResponse = best.section || aiResponse;
      } else if (response.data?.result || response.data?.answer) {
        aiResponse = response.data.result || response.data.answer;
      }

      setMessages((prev) => [...prev, { text: aiResponse, isUser: false }]);

      if (messages.length === 0) {
        const newTitle =
          input.length > 30 ? `${input.substring(0, 30)}...` : input;
        const updatedHistory = chatHistory.map((chat) =>
          chat.id === currentChatId ? { ...chat, title: newTitle } : chat
        );
        setChatHistory(updatedHistory);
        saveChatHistory(updatedHistory);
      }
    } catch (error) {
      console.error("API Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          text: "Sorry, I encountered an error. Please try again.",
          isUser: false,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  
const currentChat = chatHistory.find(chat => chat.id === currentChatId);
const currentChatTitle = currentChat ? currentChat.title : "AI Chat Assistant";

  return (
    <div className="min-h-screen bg-[#4e5561] flex">
      <ChatHistorySidebar
        chatHistory={chatHistory}
        currentChatId={currentChatId}
        showHistory={showHistory}
        chatToDelete={chatToDelete}
        onCreateNewChat={createNewChat}
        onLoadChat={loadChat}
        onSetChatToDelete={setChatToDelete}
        onDeleteChat={deleteChat}
      />

      <div className="flex-1 flex flex-col">
        <ChatHeader 
          onToggleHistory={() => setShowHistory(!showHistory)} 
          currentChatTitle={currentChatTitle} />
        
        <MessageList messages={messages} messagesEndRef={messagesEndRef} />
        
        <ChatInput
          input={input}
          loading={loading}
          onInputChange={(e) => setInput(e.target.value)}
          onSend={handleSend}
        />
      </div>
    </div>
  );
};

export default ChatbotPage;