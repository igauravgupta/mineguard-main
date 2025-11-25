import React from "react";

const ChatHistorySidebar = ({
  chatHistory,
  currentChatId,
  showHistory,
  chatToDelete,
  onCreateNewChat,
  onLoadChat,
  onSetChatToDelete,
  onDeleteChat,
}) => {
  // Get user data from localStorage or use default
  const getUserData = () => {
    const userData = localStorage.getItem("userData");
    if (userData) {
      return JSON.parse(userData);
    }
    return { username: "User", email: "user@example.com" };
  };

  const userData = getUserData();
  const username = userData.username || userData.email || "User";
  const userInitial = username.charAt(0).toUpperCase();

  return (
    <div
      className={`fixed inset-y-0 left-0 transform ${
        showHistory ? "translate-x-0" : "-translate-x-full"
      } w-64 bg-[#1F2937] shadow-lg z-10 transition-transform duration-300 md:relative md:translate-x-0 border-r border-gray-600`}
    >
      {/* Header - deepseek */}
      <div className="p-4 border-b border-gray-600">
        <h1 className="text-xl font-bold text-white">mineguard</h1>
      </div>

      {/* New Chat Button */}
      <div className="p-3 border-b border-gray-600">
  <button
    onClick={onCreateNewChat}
    className="w-full p-3 bg-[#7d828c] text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2 border border-gray-600"
  >
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
    New chat
  </button>
</div>

      {/* Chat List - Simple without days grouping */}
      <div className="overflow-y-auto h-[calc(100vh-180px)]">
        {chatHistory.map((chat) => (
          <div
            key={chat.id}
            className={`group relative p-3  cursor-pointer ${
              currentChatId === chat.id ? "bg-gray-700" : "hover:bg-gray-700"
            }`}
            onClick={() => onLoadChat(chat.id)}
          >
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-white truncate">
                  {chat.title}
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onSetChatToDelete(chat.id);
                }}
                className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-gray-200 p-1 ml-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>

            {chatToDelete === chat.id && (
              <div className="absolute right-0 top-8 bg-gray-800 shadow-lg rounded-md p-2 z-20 border border-gray-600">
                <p className="text-sm mb-2 text-white">Delete this chat?</p>
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onSetChatToDelete(null);
                    }}
                    className="text-xs px-2 py-1 border border-gray-600 rounded text-white"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteChat(chat.id);
                    }}
                    className="text-xs px-2 py-1 bg-red-500 text-white rounded"
                  >
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatHistorySidebar;