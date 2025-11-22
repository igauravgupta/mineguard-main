import DashboardCard from "../../components/website/dasboard/DashboardCard";
import Navbar from "../../components/website/dasboard/Navbar";
import { motion as Motion } from "framer-motion";

const Dashboard = () => {
  return (
    <>
      <Navbar />
      <section className="bg-gray-800 min-h-screen py-10 px-6">
        <Motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl font-semibold text-white mb-6">
            Welcome to Your Dashboard
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <DashboardCard
              title="👤 Profile"
              description="View & edit your profile."
              to="/dashboard/profile"
            />
            <DashboardCard
              title="💬 Chatbot Analytics"
              description="View usage and performance analytics for the chatbot."
              to="/dashboard/chatbot"
            />
          </div>
        </Motion.div>
      </section>
    </>
  );
};

export default Dashboard;
