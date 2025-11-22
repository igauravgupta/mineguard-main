import React from "react";
import { useNavigate } from "react-router";

const ChatDemo = () => {
  const navigate = useNavigate();
  return (
    <>
      {/* Chat Demo Section */}
      <section className="flex justify-center items-center bg-gray-900 py-20 px-4">
        <div className="flex flex-col justify-between w-full max-w-4xl h-[65vh] bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-300 dark:border-gray-700 overflow-hidden">
          {/* Header */}
          <header className="flex justify-between items-center px-5 py-3 bg-gray-100 dark:bg-gray-700 border-b border-gray-300 dark:border-gray-600">
            <div className="flex items-center gap-2 text-gray-700 dark:text-gray-200 font-semibold">
              <i className="fas fa-robot text-indigo-500"></i>
              <span>MineGuard Chatbot</span>
            </div>
            <div className="text-gray-500 dark:text-gray-300 hover:text-gray-700 cursor-pointer transition">
              <i className="fas fa-cog"></i>
            </div>
          </header>

          {/* Chat area */}
          <main className="flex-1 overflow-y-auto p-5 bg-[url('https://transparenttextures.com/patterns/cubes.png')] bg-repeat bg-opacity-10">
            {/* Left message (Bot) */}
            <div className="flex items-end gap-3 mb-6">
              <div
                className="w-11 h-11 rounded-full bg-cover bg-center"
                style={{
                  backgroundImage: `url('https://cdn-icons-png.flaticon.com/512/4712/4712104.png')`,
                }}
              />
              <div className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-100 p-4 rounded-2xl rounded-bl-none max-w-md shadow-sm">
                <div className="flex justify-between items-center text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <span>BOT</span>
                  <span>12:45</span>
                </div>
                <p>
                  Hi, I’m{" "}
                  <span className="font-semibold text-indigo-500">
                    MineGuard
                  </span>{" "}
                  👷‍♂️ — your safety assistant. How can I help you ensure
                  compliance today?
                </p>
              </div>
            </div>

            {/* Right message (User) */}
            <div className="flex items-end gap-3 justify-end mb-4">
              <div className="bg-indigo-600 text-white p-4 rounded-2xl rounded-br-none max-w-md shadow-sm">
                <div className="flex justify-between items-center text-xs text-indigo-200 mb-1">
                  <span>Gaurav</span>
                  <span>12:46</span>
                </div>
                <p>
                  What are the safety protocols for operating heavy machinery in
                  Zone 3?
                </p>
              </div>
              <div
                className="w-11 h-11 rounded-full bg-cover bg-center"
                style={{
                  backgroundImage: `url('https://cdn-icons-png.flaticon.com/512/145/145867.png')`,
                }}
              />
            </div>
          </main>

          {/* Input area */}
          <form className="flex items-center gap-3 p-4 bg-gray-100 dark:bg-gray-700 border-t border-gray-300 dark:border-gray-600">
            <input
              type="text"
              placeholder="Type your message..."
              className="flex-1 bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 px-4 py-2.5 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder:text-gray-500 dark:placeholder:text-gray-400 transition"
            />
            <button
              type="submit"
              onClick={(e) => {
                e.preventDefault();
                navigate("/login");
              }}
              className="bg-green-500 hover:bg-green-600 text-white font-semibold px-5 py-2.5 rounded-md transition"
            >
              Send
            </button>
          </form>
        </div>
      </section>

      {/* Description Section */}
      <section class="bg-white dark:bg-gray-900">
        <div class="py-8 px-4 mx-auto max-w-screen-xl lg:py-16 lg:px-6">
          <div class="max-w-screen-lg text-gray-500 sm:text-lg dark:text-gray-400">
            <h2 class="mb-4 text-4xl tracking-tight font-bold text-gray-900 dark:text-white">
              Powering innovation at{" "}
              <span class="font-extrabold">200,000+</span> companies worldwide
            </h2>
            <p class="mb-4 font-light">
              Track work across the enterprise through an open, collaborative
              platform. Link issues across Jira and ingest data from other
              software development tools, so your IT support and operations
              teams have richer contextual information to rapidly respond to
              requests, incidents, and changes.
            </p>
            <p class="mb-4 font-medium">
              Deliver great service experiences fast - without the complexity of
              traditional ITSM solutions.Accelerate critical development work,
              eliminate toil, and deploy changes with ease.
            </p>
            <a
              href="/login"
              class="inline-flex items-center font-medium text-primary-600 hover:text-primary-800 dark:text-primary-500 dark:hover:text-primary-700"
            >
              Try Chatbot
              <svg
                class="ml-1 w-6 h-6"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fill-rule="evenodd"
                  d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                  clip-rule="evenodd"
                ></path>
              </svg>
            </a>
          </div>
        </div>
      </section>
    </>
  );
};

export default ChatDemo;
