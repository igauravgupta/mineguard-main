import React from "react";

const IncidentSteps = () => {
  return (
    <section className="bg-white dark:bg-gray-900 py-20 px-4">
      <div className="mx-auto max-w-screen-md">
        {/* Section Header */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-extrabold tracking-tight text-gray-900 dark:text-white">
            Incident Reporting Process
          </h2>
          <p className="mt-4 text-lg text-gray-500 dark:text-gray-400">
            Follow these five simple steps to report an incident through{" "}
            <span className="font-semibold text-indigo-600 dark:text-indigo-400">
              MineGuard
            </span>
            . Our system ensures quick response, accurate tracking, and safety
            compliance.
          </p>
        </div>

        {/* Timeline */}
        <ol className="relative border-s border-gray-200 dark:border-gray-700">
          {/* Step 1 */}
          <li className="mb-10 ms-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -start-1.5 border border-white dark:border-gray-900 dark:bg-gray-700"></div>
            <time className="mb-1 text-sm font-normal leading-none text-gray-400 dark:text-gray-500">
              Step 1
            </time>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Identify and Describe the Incident
            </h3>
            <p className="mb-4 text-base font-normal text-gray-500 dark:text-gray-400">
              Start by providing details about what happened — when, where, and
              the type of incident (e.g., safety hazard, injury, or near miss).
              Include as much information as possible to help the system
              understand the situation.
            </p>
            <a
              href="#"
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-900 bg-white border border-gray-200 rounded-lg hover:bg-gray-100 hover:text-indigo-600 focus:z-10 focus:ring-4 focus:outline-none focus:ring-gray-100 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700 dark:focus:ring-gray-700"
            >
              Learn more
              <svg
                className="w-3 h-3 ms-2 rtl:rotate-180"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 14 10"
              >
                <path
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M1 5h12m0 0L9 1m4 4L9 9"
                />
              </svg>
            </a>
          </li>

          {/* Step 2 */}
          <li className="mb-10 ms-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -start-1.5 border border-white dark:border-gray-900 dark:bg-gray-700"></div>
            <time className="mb-1 text-sm font-normal leading-none text-gray-400 dark:text-gray-500">
              Step 2
            </time>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Upload Supporting Evidence
            </h3>
            <p className="text-base font-normal text-gray-500 dark:text-gray-400">
              Attach relevant photos, videos, or documents. These help validate
              your report and allow safety teams to assess the severity and root
              cause efficiently.
            </p>
          </li>

          {/* Step 3 */}
          <li className="mb-10 ms-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -start-1.5 border border-white dark:border-gray-900 dark:bg-gray-700"></div>
            <time className="mb-1 text-sm font-normal leading-none text-gray-400 dark:text-gray-500">
              Step 3
            </time>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              AI Automatically Categorizes the Event
            </h3>
            <p className="text-base font-normal text-gray-500 dark:text-gray-400">
              The system’s AI analyzes your input and classifies the incident
              based on type, severity, and potential impact, ensuring it reaches
              the right department immediately.
            </p>
          </li>

          {/* Step 4 */}
          <li className="mb-10 ms-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -start-1.5 border border-white dark:border-gray-900 dark:bg-gray-700"></div>
            <time className="mb-1 text-sm font-normal leading-none text-gray-400 dark:text-gray-500">
              Step 4
            </time>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Notification Sent to Responsible Authority
            </h3>
            <p className="text-base font-normal text-gray-500 dark:text-gray-400">
              Once categorized, the report is automatically sent to your
              organization’s designated authority for quick action and follow-up.
            </p>
          </li>

          {/* Step 5 */}
          <li className="ms-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -start-1.5 border border-white dark:border-gray-900 dark:bg-gray-700"></div>
            <time className="mb-1 text-sm font-normal leading-none text-gray-400 dark:text-gray-500">
              Step 5
            </time>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Track Status and Updates
            </h3>
            <p className="text-base font-normal text-gray-500 dark:text-gray-400">
              Stay informed with real-time updates. Check your report’s progress,
              corrective actions, and final resolution through the MineGuard chatbot.
            </p>
          </li>
        </ol>
      </div>
    </section>
  );
};

export default IncidentSteps;
