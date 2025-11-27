import React from "react";

const ChatInput = ({ input, loading, onInputChange, onSend }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    onSend(e);
  };

  return (
    <div className="p-4 border-t border-gray-700 bg-gray-900">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={onInputChange}
          placeholder="Type your message..."
          className="flex-1 p-3 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-500 bg-gray-800 text-white placeholder-gray-400"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="px-4 py-2 bg-[#4e5561] text-white rounded-lg hover:bg-gray-700 disabled:opacity-50"
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
};

export default ChatInput;