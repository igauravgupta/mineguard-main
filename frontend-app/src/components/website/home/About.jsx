// src/pages/AboutPage.js
import Feature from "./Feature";

const AboutPage = () => {
  return (
    <div>
      <section className="bg-white dark:bg-gray-900">
        <div className="gap-16 items-center py-8 px-4 mx-auto max-w-screen-xl lg:grid lg:grid-cols-2 lg:py-16 lg:px-6">
          <div className="font-light text-gray-500 sm:text-lg dark:text-gray-400">
            <h2 className="mb-4 text-4xl tracking-tight font-extrabold text-gray-900 dark:text-white">
              About MineGuard
            </h2>
            <p className="mb-4">
              MineGuard is an AI-powered platform designed to make mining operations safer and more compliant. We provide instant answers to mining regulations through our smart chatbot and enable quick digital reporting of safety incidents. Our goal is to help mining companies follow rules easily while protecting their workers.
            </p>
            <p>
              We combine technology with safety to create simple solutions for complex mining challenges. MineGuard ensures that every mining professional has the right tools to work safely and stay compliant with regulations.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4 mt-3">
            <img
              className="w-full rounded-lg"
              src="https://plus.unsplash.com/premium_photo-1677707394192-51c2409af494?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
              alt="office content 1"
            />
            <img
              className="mt-4 w-full lg:mt-10 rounded-lg"
              src="https://images.unsplash.com/photo-1685698426903-15f4c4d397fd?q=80&w=694&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
              alt="office content 2"
            />
          </div>
        </div>
      </section>
    </div>
  );
};

export default AboutPage;
