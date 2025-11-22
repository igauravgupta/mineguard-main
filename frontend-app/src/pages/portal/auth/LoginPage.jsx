import Navbar from "../../../components/portal/auth/Navbar";
import LoginForm from "../../../components/portal/auth/LoginForm";

const LoginPage = () => {
  return (
    <div className="bg-gray-900 min-h-screen">
      <Navbar />
      <LoginForm
        redirect="/portal/user/dashboard"
        signIn="/portal/user/signup"
      />
    </div>
  );
};

export default LoginPage;
