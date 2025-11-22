// src/pages/Home.jsx
import Navbar from "../../components/website/chatbot/Navbar";
import Footer from "../../components/website/shared/Footer";
import About from "../../components/website/chatbot/About";
import ChatDemo from "../../components/website/chatbot/ChatDemo";
const Home = () => {
  return (
    <div className="min-h-screen flex flex-col justify-between scroll-smooth">
      <Navbar />
      <About />
      <ChatDemo />
      <Footer />
    </div>
  );
};

export default Home;
