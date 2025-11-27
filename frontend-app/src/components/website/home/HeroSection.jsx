import { motion as Motion } from "framer-motion";

const HeroSection = () => {
  return (
    <main className="relative bg-gray-900 overflow-hidden flex items-center justify-center pt-60 pb-20">
      <div className="relative z-10 flex flex-col items-center justify-center px-6">
        <Motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="w-full flex flex-col items-center relative"
        >
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold text-center text-slate-100 leading-tight tracking-tight">
            <span className="block">MineGuard</span>
            <span className="block text-3xl md:text-4xl text-slate-200 font-semibold mt-3">
              A Comprehensive Solution for Mining Operations
            </span>
          </h1>

          <p className="text-sm md:text-base text-slate-400 mt-6 max-w-2xl text-center mb-2">
            Offers regulatory guidance, on-site reporting, and compliance
            oversight.
          </p>

          <div className="flex items-center gap-4 mt-3">
            <Motion.a
              href="/dashboard/chatbot"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="inline-flex items-center gap-2 px-4 py-3 rounded-md bg-gradient-to-r from-violet-600 to-purple-500 text-white font-semibold shadow-lg focus:outline-none focus:ring-4 focus:ring-violet-400/25"
            >
              Get started. It's Live!
            </Motion.a>

            <Motion.a
              href="#"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              className="inline-flex items-center gap-2 px-4 py-3 rounded-md bg-white/95 text-slate-800 border border-slate-200 shadow-sm hover:shadow-md focus:outline-none"
            >
              <svg
                className="w-4 h-4 text-slate-700"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth="2"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M16 8l-4-4-4 4"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 4v12"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"
                />
              </svg>
              Explore more
            </Motion.a>
          </div>
        </Motion.div>
      </div>
    </main>
  );
};

export default HeroSection;
