import { ChevronDownIcon } from "@heroicons/react/16/solid";

export default function Contact() {
  return (
    <section
      id="contact"
      className="relative isolate bg-gray-900 px-6 py-24 sm:py-32 lg:px-8 overflow-hidden"
    >
      <div className="mx-auto max-w-7xl grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
        {/* LEFT SIDE - HEADING + EXTRA CONTENT */}
        <div className="max-w-xl">
          <h2 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-left text-center">
            Let’s Build Something Together
          </h2>
          <p className="mt-6 text-lg text-gray-400 leading-relaxed lg:text-left text-center">
            Have a question, project idea, or partnership opportunity? We’d love
            to hear from you. Reach out and our team will respond within 24
            hours.
          </p>

          {/* Extra content / Contact info */}
          <div className="mt-10 space-y-6 text-gray-300">
            <div>
              <h4 className="text-white font-semibold">📍 Office Address</h4>
              <p className="text-gray-400">
                123 Innovation Drive, Tech City, NY 10001
              </p>
            </div>
            <div>
              <h4 className="text-white font-semibold">📞 Phone</h4>
              <p className="text-gray-400">+1 (555) 987-6543</p>
            </div>
            <div>
              <h4 className="text-white font-semibold">📧 Email</h4>
              <p className="text-gray-400">support@yourcompany.com</p>
            </div>
          </div>

          {/* Optional CTA */}
          <div className="mt-12">
            <p className="text-gray-400 italic">
              “Your vision, our expertise — let’s make it happen.”
            </p>
          </div>
        </div>

        {/* RIGHT SIDE - FORM */}
        <form
          action="#"
          method="POST"
          className="bg-gray-800/60 backdrop-blur-sm rounded-2xl border border-gray-700 p-8 shadow-xl transition-all hover:shadow-2xl"
        >
          <div className="grid grid-cols-1 gap-x-8 gap-y-6 sm:grid-cols-2">
            {/* First Name */}
            <div>
              <label
                htmlFor="first-name"
                className="block text-sm font-semibold text-gray-200"
              >
                First name
              </label>
              <input
                id="first-name"
                name="first-name"
                type="text"
                placeholder="John"
                className="mt-2.5 block w-full rounded-md bg-gray-900/50 px-3.5 py-2 text-white border border-gray-700 placeholder:text-gray-500 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 transition"
              />
            </div>

            {/* Last Name */}
            <div>
              <label
                htmlFor="last-name"
                className="block text-sm font-semibold text-gray-200"
              >
                Last name
              </label>
              <input
                id="last-name"
                name="last-name"
                type="text"
                placeholder="Doe"
                className="mt-2.5 block w-full rounded-md bg-gray-900/50 px-3.5 py-2 text-white border border-gray-700 placeholder:text-gray-500 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 transition"
              />
            </div>

            {/* Email */}
            <div className="sm:col-span-2">
              <label
                htmlFor="email"
                className="block text-sm font-semibold text-gray-200"
              >
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                placeholder="you@example.com"
                className="mt-2.5 block w-full rounded-md bg-gray-900/50 px-3.5 py-2 text-white border border-gray-700 placeholder:text-gray-500 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 transition"
              />
            </div>

            {/* Phone */}
            <div className="sm:col-span-2">
              <label
                htmlFor="phone-number"
                className="block text-sm font-semibold text-gray-200"
              >
                Phone number
              </label>
              <div className="mt-2.5 flex items-center rounded-md bg-gray-900/50 border border-gray-700 focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-500 transition">
                <div className="relative">
                  <select
                    id="country"
                    name="country"
                    className="appearance-none bg-transparent py-2 pr-8 pl-3.5 text-sm text-gray-400 focus:outline-none"
                  >
                    <option>US</option>
                    <option>CA</option>
                    <option>EU</option>
                  </select>
                  <ChevronDownIcon
                    aria-hidden="true"
                    className="pointer-events-none absolute right-2 top-2.5 w-4 h-4 text-gray-400"
                  />
                </div>
                <input
                  id="phone-number"
                  name="phone-number"
                  type="text"
                  placeholder="123-456-7890"
                  className="w-full bg-transparent py-2 pr-3 pl-3 text-white placeholder:text-gray-500 focus:outline-none"
                />
              </div>
            </div>

            {/* Message */}
            <div className="sm:col-span-2">
              <label
                htmlFor="message"
                className="block text-sm font-semibold text-gray-200"
              >
                Message
              </label>
              <textarea
                id="message"
                name="message"
                rows={4}
                placeholder="Write your message..."
                className="mt-2.5 block w-full rounded-md bg-gray-900/50 px-3.5 py-2 text-white border border-gray-700 placeholder:text-gray-500 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 transition"
              />
            </div>

            {/* Checkbox */}
            <div className="flex items-start gap-3 sm:col-span-2">
              <input
                id="agree"
                name="agree"
                type="checkbox"
                className="mt-1 h-4 w-4 rounded border-gray-700 bg-gray-900/50 text-indigo-500 focus:ring-indigo-500"
              />
              <label htmlFor="agree" className="text-sm text-gray-400">
                By selecting this, you agree to our{" "}
                <a href="#" className="text-indigo-400 hover:text-indigo-300">
                  privacy policy
                </a>
                .
              </label>
            </div>
          </div>

          {/* Submit Button */}
          <div className="mt-10">
            <button
              type="submit"
              className="block w-full rounded-md bg-gradient-to-r from-indigo-500 to-violet-600 px-3.5 py-3 text-center text-sm font-semibold text-white shadow-md hover:from-indigo-400 hover:to-violet-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 transition-all"
            >
              Let’s Talk
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
