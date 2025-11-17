import logo from "../../../assets/logo.png";
const Navbar = () => {
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
        </nav>
      </header>
    </div>
  );
};

export default Navbar;
