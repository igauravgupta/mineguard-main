import { useState, useEffect, useRef } from "react";
import logo from "../../../assets/logo.png";
// import { useNavigate } from "react-router-dom";
import {
  SignedIn,
  SignedOut,
  SignInButton,
  UserButton,
} from "@clerk/clerk-react";
const navigation = [
  { name: "About", targetId: "about" },
  { name: "Features", targetId: "features" },
  {
    name: "Solutions",
    children: [
      { name: "Chatbot", href: "/chatbot" },
      { name: "Incident Reporting", href: "/incident-reporting" },
    ],
  },
  { name: "Developers", targetId: "team" },
  { name: "Contact", targetId: "contact" },
];

const Navbar = () => {
  const [openDropdown, setOpenDropdown] = useState(null);
  const dropdownRef = useRef(null);
  // const navigate = useNavigate();

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setOpenDropdown(null);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggleDropdown = (name) => {
    setOpenDropdown(openDropdown === name ? null : name);
  };

  // Smooth scroll to section
  const handleScroll = (targetId) => {
    if (!targetId) return;
    const section = document.getElementById(targetId);
    if (section) {
      section.scrollIntoView({ behavior: "smooth" });
      setOpenDropdown(null);
    }
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
                          onClick={() => setOpenDropdown(null)} // close when a link is clicked
                          className="block px-4 py-2 text-sm hover:bg-gray-800 transition-colors"
                        >
                          {child.name}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <button
                  key={item.name}
                  onClick={() => handleScroll(item.targetId)}
                  className="text-sm font-semibold text-white hover:text-gray-300 focus:outline-none"
                >
                  {item.name}
                </button>
              )
            )}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-4">
            <SignedOut>
              <SignInButton forceRedirectUrl="/dashboard/chatbot">
                <button className="text-sm font-semibold text-white hover:text-gray-300 focus:outline-none">
                  Get Started
                </button>
              </SignInButton>
            </SignedOut>

            <SignedIn>
              {/* Dashboard nav item first, then UserButton */}
              <a
                href="/dashboard"
                className="text-sm font-semibold text-white hover:text-gray-300 focus:outline-none"
              >
                Dashboard
              </a>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
          </div>
        </nav>
      </header>
    </div>
  );
};

export default Navbar;
