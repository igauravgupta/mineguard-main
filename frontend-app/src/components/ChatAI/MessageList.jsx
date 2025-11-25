import React from "react";
import ReactMarkdown from "react-markdown";

const MessageList = ({ messages, messagesEndRef }) => {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-900">
      {messages.length === 0 ? (
        <div className="h-full flex items-center justify-center text-gray-300">
          <p className="text-2xl">Hi !  How can I help you today?</p>
        </div>
      ) : (
        messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.isUser ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                msg.isUser
                  ? "bg-[#4e5561] text-white rounded-br-none"
                  : "bg-gray-800 text-white rounded-bl-none"
              }`}
            >
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            </div>
          </div>
        ))
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;