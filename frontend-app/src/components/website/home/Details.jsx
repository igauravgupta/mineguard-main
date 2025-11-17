const Details = () => {
  return (
    <section className="relative bg-gray-900 py-24 overflow-hidden">
      {/* Subtle gradient background */}
      {/* <div
        aria-hidden="true"
        className="absolute inset-0 bg-gradient-to-br from-indigo-900/40 via-gray-900 to-gray-950"
      ></div> */}

      <div className="relative w-full max-w-7xl px-6 md:px-8 mx-auto">
        <div className="grid lg:grid-cols-2 items-center gap-16">
          {/* IMAGE COLUMN */}
          <div className="flex justify-center lg:justify-start order-1">
            <div className="relative rounded-3xl overflow-hidden border border-gray-700 bg-gray-800/50 backdrop-blur-sm shadow-xl hover:shadow-2xl transition-all duration-700 w-full max-w-[560px]">
              <img
                className="w-full h-full object-cover scale-105 hover:scale-110 transition-transform duration-700"
                src="https://pagedone.io/asset/uploads/1717742431.png"
                alt="about us"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-gray-900/60 via-transparent to-transparent" />
            </div>
          </div>

          {/* CONTENT COLUMN */}
          <div className="flex flex-col justify-center items-start gap-10 order-2">
            {/* Header */}
            <div className="flex flex-col items-start gap-3">
              <h2 className="text-white text-4xl sm:text-5xl font-bold leading-snug">
                The Tale of Our{" "}
                <span className="text-indigo-500">Achievement</span> Story
              </h2>
              <p className="text-gray-400 text-base leading-relaxed max-w-xl">
                Our achievement story is a testament to teamwork and
                perseverance. Together, we've overcome challenges, celebrated
                victories, and created a narrative of progress and success.
              </p>
            </div>

            {/* Stats Section */}
            <div className="grid sm:grid-cols-2 gap-6 w-full">
              {[
                {
                  title: "33+ Years",
                  desc: "Influencing Digital Landscapes Together",
                },
                {
                  title: "125+ Projects",
                  desc: "Excellence Achieved Through Success",
                },
                {
                  title: "26+ Awards",
                  desc: "Our Dedication to Innovation Wins Understanding",
                },
                {
                  title: "99% Happy Clients",
                  desc: "Mirrors our Focus on Client Satisfaction",
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
            <button className="mt-4 flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-lg font-medium shadow-md hover:shadow-indigo-500/20 hover:scale-105 transition-all duration-300">
              Read More
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="18"
                height="18"
                fill="none"
                viewBox="0 0 18 18"
                className="transition-transform duration-300 group-hover:translate-x-1"
              >
                <path
                  d="M6.75265 4.49658L11.2528 8.99677L6.75 13.4996"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Details;
