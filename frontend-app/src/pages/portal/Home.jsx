import Navbar from "../../components/portal/auth/Navbar";
import Footer from "../../components/website/shared/Footer";
import { useNavigate } from "react-router-dom";

const PortalHome = () => {
  const navigate = useNavigate();

  return (
    <section className="bg-gray-900 min-h-screen">
      <Navbar />
      <div className="-[#0f0f0f] flex flex-col items-center justify-center text-center text-white pt-40 pb-40">
        {/* Heading */}
        <h1 className="text-4xl font-bold mb-3">Choose Your Portal</h1>
        <p className="text-gray-400 mb-10 max-w-xl">
          Select the appropriate portal to continue. Access admin tools or
          submit incident reports.
        </p>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl w-full">
          {/* Admin Portal Card */}
          <div className="bg-gray-800 border border-gray-800 rounded-xl shadow-lg hover:shadow-purple-500/20 transition-all duration-300 p-8 flex flex-col items-center">
            <h2 className="text-2xl font-semibold mb-4">Admin Portal</h2>
            <p className="text-gray-400 mb-8 text-sm">
              Manage incidents, monitor reports, and view analytics in one
              dashboard.
            </p>
            <button
              onClick={() => navigate("/portal/admin/login")}
              className="w-full py-2 rounded-lg bg-purple-600 hover:bg-purple-700 transition-all font-medium"
            >
              Go to Admin Portal
            </button>
          </div>

          {/* User Portal Card */}
          <div className="bg-gray-800 border border-gray-800 rounded-xl shadow-lg hover:shadow-purple-500/20 transition-all duration-300 p-8 flex flex-col items-center">
            <h2 className="text-2xl font-semibold mb-4">User Portal</h2>
            <p className="text-gray-400 mb-8 text-sm">
              Report incidents, chat with AI assistant, and track updates
              easily.
            </p>
            <button
              onClick={() => navigate("/portal/user/login")}
              className="w-full py-2 rounded-lg bg-purple-600 hover:bg-purple-700 transition-all font-medium"
            >
              Go to User Portal
            </button>
          </div>
        </div>
      </div>
      <Footer />
    </section>
  );
};

export default PortalHome;
