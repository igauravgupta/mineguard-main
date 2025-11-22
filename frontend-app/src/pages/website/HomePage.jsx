// src/pages/Home.jsx
import Navbar from "../../components/website/home/Navbar";
import Footer from "../../components/website/shared/Footer";
import HeroSection from "../../components/website/home/HeroSection";
import About from "../../components/website/home/About";
import Contact from "../../components/website/home/Contact";
import TeamPage from "../../components/website/home/Team";
import GridPattern from "../../components/website/home/GridPattern";
import Details from "../../components/website/home/Details";
import { useState, useEffect } from "react";

const Home = () => {

    const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    const observer = new MutationObserver(() => {
      const modal = document.querySelector('.cl-modal, [data-cl-component="Modal"]');
      setIsModalOpen(!!modal);
    });

    // Observe full document
    observer.observe(document.body, { childList: true, subtree: true });
    
    return () => observer.disconnect();
  }, []);

  return (
   <div className="min-h-screen flex flex-col justify-between scroll-smooth">
      <div className={`${isModalOpen ? "blur-sm transition-all duration-300" : ""}`}>
        <Navbar />
        <main className={`${isModalOpen ? "blur-sm transition-all duration-300" : ""}`}>
          <section id="home"><HeroSection /></section>
          <section id="about"><About /></section>
          <section id="features"><GridPattern /></section>
          <section id="details"><Details /></section>
          <section id="team"><TeamPage /></section>
          <section id="contact"><Contact /></section>
          <Footer />
        </main>
      </div>
    </div>

  );
};

export default Home;
