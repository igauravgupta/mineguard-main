import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

const LoginForm = ({ redirect, signIn }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    alert(`Logged in as ${username}`);
    navigate(redirect);
  };

  const handleGoogleLogin = () => {
    alert(`Logging in with Google`);
    navigate(redirect);
  };

  return (
    <section className="flex items-center justify-center bg-gray-100 dark:bg-gray-900 px-4 pt-40">
      <div className="w-full max-w-md bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 border border-gray-200 dark:border-gray-700 transition-all duration-300 hover:shadow-purple-500/10">
        {/* Title */}
        <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-white mb-6">
          Welcome Back to{" "}
          <span className="text-indigo-600 dark:text-indigo-400">
            MineGuard
          </span>
        </h2>

        {/* Form */}
        <form onSubmit={handleLogin} className="space-y-6">
          {/* Username */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-2">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="Enter your username"
              required
            />
          </div>

          {/* Password */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-2">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="Enter your password"
              required
            />
          </div>

          {/* Login Button */}
          <button
            type="submit"
            className="w-full py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold rounded-md transition-all"
          >
            Login
          </button>

          {/* Divider */}
          <div className="flex items-center justify-center my-4">
            <div className="border-t border-gray-300 dark:border-gray-600 flex-grow"></div>
            <span className="mx-2 text-gray-500 dark:text-gray-400 text-sm">
              OR
            </span>
            <div className="border-t border-gray-300 dark:border-gray-600 flex-grow"></div>
          </div>

          {/* Google Login */}
          <button
            type="button"
            onClick={handleGoogleLogin}
            className="w-full flex items-center justify-center gap-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-200 bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 transition-all"
          >
            <img
              src="https://www.svgrepo.com/show/355037/google.svg"
              alt="Google"
              className="w-5 h-5"
            />
            <span>Login with Google</span>
          </button>
        </form>

        {/* Signup Link */}
        <p className="text-sm text-center text-gray-500 dark:text-gray-400 mt-6">
          Don’t have an account?{" "}
          <Link
            to={signIn}
            className="text-indigo-600 dark:text-indigo-400 hover:underline"
          >
            Sign up
          </Link>
        </p>
      </div>
    </section>
  );
};

export default LoginForm;
