

const Details = () => {
  return (
    <section className="relative bg-gray-900 py-24 overflow-hidden">
      <div className="relative w-full max-w-7xl px-6 md:px-8 mx-auto">
        <div className="grid lg:grid-cols-2 items-center gap-16">
          {/* IMAGE COLUMN */}
          <div className="flex justify-center lg:justify-start order-1">
            <div className="relative rounded-3xl overflow-hidden border border-gray-700 bg-gray-800/50 backdrop-blur-sm shadow-xl hover:shadow-2xl transition-all duration-700 w-full max-w-[560px]">
              <img
                className="w-full h-full object-cover scale-105 hover:scale-110 transition-transform duration-700"
                src="https://images.unsplash.com/photo-1581091226033-d5c48150dbaa?ixlib=rb-4.0.1&auto=format&fit=crop&w=1000&q=80"
                alt="MineGuard Safety Platform"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-gray-900/60 via-transparent to-transparent" />
            </div>
          </div>

          {/* CONTENT COLUMN */}
          <div className="flex flex-col justify-center items-start gap-10 order-2">
            {/* Header */}
            <div className="flex flex-col items-start gap-3">
              <h2 className="text-white text-4xl sm:text-5xl font-bold leading-snug">
                Revolutionizing Mining{" "}
                <span className="text-indigo-500">Safety & Compliance</span>
              </h2>
              <p className="text-gray-400 text-base leading-relaxed max-w-xl">
                MineGuard combines cutting-edge AI technology with deep industry expertise to transform how mining companies handle safety and regulatory compliance. Our platform makes complex regulations accessible and safety reporting effortless.
              </p>
            </div>

            {/* Stats Section */}
            <div className="grid sm:grid-cols-2 gap-6 w-full">
              {[
                {
                  title: "5000+ Regulations",
                  desc: "Instant access to comprehensive mining laws and rules",
                },
                {
                  title: "24/7 Availability",
                  desc: "Round-the-clock access to safety information and reporting",
                },
                {
                  title: "Digital Reporting", 
                  desc: "Streamlined incident and hazard reporting system",
                },
                {
                  title: "Cost Effective",
                  desc: "Reduce compliance costs and penalty risks",
                },
              ].map((stat, idx) => (
                <div
                  key={idx}
                  className="p-5 rounded-xl bg-gray-800/60 border border-gray-700 hover:border-indigo-500 transition-all duration-500 hover:shadow-md"
                >
                  <h4 className="text-white text-2xl font-semibold mb-1">
                    {stat.title}
                  </h4>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    {stat.desc}
                  </p>
                </div>
              ))}
            </div>

            {/* Button */}

            <div className="mt-6 flex flex-wrap gap-4">
              {/* Chatbot Button */}
              <button 
                onClick={() => {
                  // Navigate to chatbot interface
                  // Or open chatbot modal
                  window.location.href = '/chatbot';
                }}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-lg font-medium shadow-md hover:shadow-indigo-500/20 hover:scale-105 transition-all duration-300"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                Ask Regulations
              </button>

              {/* Incident Reporting Button */}
              <button 
                onClick={() => {
                  // Navigate to reporting interface
                  window.location.href = '/incident-reporting';
                }}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg font-medium shadow-md hover:shadow-green-500/20 hover:scale-105 transition-all duration-300"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                Report Incident
              </button>
            </div> 
          </div>
        </div>
      </div>
    </section>
  );
};

export default Details;