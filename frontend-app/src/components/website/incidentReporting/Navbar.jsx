import logo from "../../../assets/logo.png";
import { useNavigate } from "react-router-dom";

const Navbar = () => {
  const navigate = useNavigate();

  return (
    <div className="bg-gray-900">
      <header className="w-full z-50">
        <nav
          aria-label="Global"
          className="flex items-center justify-between p-6 lg:px-8"
        >
          {/* Logo */}
          <div className="flex">
            <a href="/" className="-m-1.5 p-1.5">
              <span className="sr-only">Your Company</span>
              <img alt="" src={logo} className="h-8 w-auto" />
            </a>
          </div>

          {/* Right side */}
          <div className="flex items-center">
            <button
              onClick={() => navigate("/login")}
              className="text-sm font-semibold text-white hover:text-gray-300 focus:outline-none"
            >
              Report Incident <span aria-hidden="true">&rarr;</span>
            </button>
          </div>
        </nav>
      </header>
    </div>
  );
};

export default Navbar;
