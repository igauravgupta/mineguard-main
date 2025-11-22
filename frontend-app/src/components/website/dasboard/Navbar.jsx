import { useState, useEffect, useRef } from "react";
import logo from "../../../assets/logo.png";
import { Link } from "react-router-dom";
import {
  SignedIn,
  SignedOut,
  SignInButton,
  UserButton,
} from "@clerk/clerk-react";
const navigation = [
  { name: "Home", to: "/" },
  { name: "About Chatbot", to: "/chatbot" },
  { name: "Reports", to: "/reports" },
];

const Navbar = () => {
  const [openDropdown, setOpenDropdown] = useState(null);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setOpenDropdown(null);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [dropdownRef]);

  const toggleDropdown = (name) => {
    setOpenDropdown(openDropdown === name ? null : name);
  };

  return (
    <div className="bg-gray-900">
      <header className="w-full z-50">
        <nav
          aria-label="Global"
          className="flex items-center justify-between p-6 lg:px-8"
        >
          {/* Logo */}
          <div className="flex">
            <a href="#" className="-m-1.5 p-1.5">
              <span className="sr-only">Your Company</span>
              <img alt="" src={logo} className="h-8 w-auto" />
            </a>
          </div>

          {/* Navigation Links */}
          <div
            ref={dropdownRef}
            className="flex gap-x-12 items-center relative"
          >
            {navigation.map((item) =>
              item.children ? (
                <div key={item.name} className="relative">
                  <button
                    type="button"
                    onClick={() => toggleDropdown(item.name)}
                    className="text-sm font-semibold text-white inline-flex items-center gap-1 focus:outline-none"
                  >
                    {item.name}
                    <svg
                      className={`ml-1 h-3 w-3 text-white transition-transform ${
                        openDropdown === item.name ? "rotate-180" : ""
                      }`}
                      viewBox="0 0 20 20"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                      aria-hidden="true"
                    >
                      <path
                        d="M6 8l4 4 4-4"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </button>

                  {/* Dropdown */}
                  {openDropdown === item.name && (
                    <div
                      className="
                        absolute left-0 mt-2 w-48 rounded-md shadow-lg py-2 z-50
                        bg-gray-900 text-white
                        transition-all duration-200 border border-gray-800
                      "
                    >
                      {item.children.map((child) => (
                        <a
                          key={child.name}
                          href={child.href}
                          onClick={() => setOpenDropdown(null)}
                          className="block px-4 py-2 text-sm hover:bg-gray-800 transition-colors"
                        >
                          {child.name}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <Link
                  key={item.name}
                  to={item.to}
                  className="text-sm font-semibold text-white hover:text-gray-300 focus:outline-none"
                >
                  {item.name}
                </Link>
              )
            )}
          </div>

          {/* Right side */}
          <div className="flex items-center">
            <SignedOut>
              <SignInButton forceRedirectUrl="/dashboard">
                <button className="text-sm font-semibold text-white hover:text-gray-300 focus:outline-none">
                  Get Started
                </button>
              </SignInButton>
            </SignedOut>

            <SignedIn>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
          </div>
        </nav>
      </header>
    </div>
  );
};

export default Navbar;
