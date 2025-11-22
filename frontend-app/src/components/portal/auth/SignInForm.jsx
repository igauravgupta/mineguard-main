import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const SignupForm = ({ redirect, logIn }) => {
  const [formData, setFormData] = useState({
    fullName: "",
    username: "",
    email: "",
    password: "",
  });

  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSignup = (e) => {
    e.preventDefault();
    navigate(redirect);
  };

  const handleGoogleSignup = () => {
    alert("Google signup is not implemented yet.");
    navigate(redirect);
  };

  return (
    <section className="flex items-center justify-center bg-gray-100 dark:bg-gray-900 px-4 pt-6">
      <div className="w-full max-w-md bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 border border-gray-200 dark:border-gray-700 transition-all duration-300 hover:shadow-purple-500/10">
        {/* Title */}
        <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-white mb-6">
          Create Your{" "}
          <span className="text-indigo-600 dark:text-indigo-400">
            MineGuard
          </span>{" "}
          Account
        </h2>

        <p className="text-center text-gray-500 dark:text-gray-400 mb-8 text-sm">
          Fill in your details or sign up with Google to continue.
        </p>

        {/* Form */}
        <form onSubmit={handleSignup} className="space-y-6">
          {/* Full Name */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-2">
              Full Name
            </label>
            <input
              type="text"
              name="fullName"
              value={formData.fullName}
              onChange={handleChange}
              required
              placeholder="John Doe"
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Username */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-2">
              Username
            </label>
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
              placeholder="johndoe"
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Email */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-2">
              Email Address
            </label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              placeholder="you@example.com"
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Password */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-2">
              Password
            </label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              placeholder="********"
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Signup Button */}
          <button
            type="submit"
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-md transition-all"
          >
            Sign Up
          </button>

          {/* Divider */}
          <div className="flex items-center justify-center my-4">
            <div className="border-t border-gray-300 dark:border-gray-600 flex-grow"></div>
            <span className="mx-2 text-gray-500 dark:text-gray-400 text-sm">
              OR
            </span>
            <div className="border-t border-gray-300 dark:border-gray-600 flex-grow"></div>
          </div>

          {/* Google Sign-in */}
          <button
            type="button"
            onClick={handleGoogleSignup}
            className="w-full flex items-center justify-center gap-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-200 bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 transition-all"
          >
            <img
              src="https://www.svgrepo.com/show/355037/google.svg"
              alt="Google"
              className="w-5 h-5"
            />
            <span>Sign up with Google</span>
          </button>
        </form>

        {/* Footer */}
        <p className="text-sm text-center text-gray-500 dark:text-gray-400 mt-6">
          Already have an account?{" "}
          <button
            onClick={() => navigate(logIn)}
            className="text-indigo-600 dark:text-indigo-400 hover:underline"
          >
            Login here
          </button>
        </p>
      </div>
    </section>
  );
};

export default SignupForm;
