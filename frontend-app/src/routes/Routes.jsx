import { Routes, Route } from "react-router-dom";
import ProtectedRoute from "./ProtectedRoute";

// Website imports
import Home from "../pages/website/HomePage";
import Chatbot from "../pages/website/ChatbotPage";
import IncidentReportingPage from "../pages/website/IncidentReportingPage";

// Dashboard
import Dashboard from "../pages/website/Dashboard";

// live-chatbot
import LiveChatbot from "../pages/website/Chatbot";

// Portal Auth Pages
import PortalHome from "../pages/portal/Home";
import LoginPageUser from "../pages/portal/auth/LoginPage";
import SignUpPageUser from "../pages/portal/auth/SignUpPage";
import LoginPageAdmin from "../pages/portal/admin/LoginPage";
import SignUpPageAdmin from "../pages/portal/admin/SignUpPage";

const AppRoutes = () => {
  return (
    <Routes>
      {/* --- Basic Website Routes --- */}
      <Route path="/" element={<Home />} />
      <Route path="/chatbot" element={<Chatbot />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/dashboard/chatbot"
        element={
          <ProtectedRoute>
            <LiveChatbot />
          </ProtectedRoute>
        }
      />

      {/* --- Portal Routes --- */}
      <Route path="/portal/home" element={<PortalHome />} />
      <Route path="/portal/user/login" element={<LoginPageUser />} />
      <Route path="/portal/user/signup" element={<SignUpPageUser />} />
      <Route path="/portal/admin/login" element={<LoginPageAdmin />} />
      <Route path="/portal/admin/signup" element={<SignUpPageAdmin />} />

      {/* --- Incident Reporting --- */}
      <Route path="/incident-reporting" element={<IncidentReportingPage />} />
    </Routes>
  );
};

export default AppRoutes;
