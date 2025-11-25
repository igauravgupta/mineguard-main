import React from "react";

const ChatHeader = ({ onToggleHistory, currentChatTitle }) => {
  return (
    <div className="bg-gray-900 p-4 flex items-center border-b border-gray-700">
      <button
        onClick={onToggleHistory}
        className="text-white mr-4 p-2 rounded-full hover:bg-gray-800 transition-colors md:hidden"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>
         <div className="flex-1 flex justify-center">
            <h1 className="text-xl font-semibold text-white text-center">
            {currentChatTitle || "AI Chat Assistant"}
            </h1>
      </div>
    </div>
  );
};

export default ChatHeader;