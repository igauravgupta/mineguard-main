const Card = ({ name, githubId, linkedinId, image }) => {
  return (
    <div className="w-full max-w-sm bg-gray-800 border border-gray-700 rounded-xl shadow-md hover:shadow-lg transition-all duration-300 overflow-hidden flex flex-col items-center py-8 px-4">
      {/* Profile Image */}
      <img
        className="w-24 h-24 mb-4 rounded-full shadow-lg object-cover border-2 border-purple-500/50"
        src={image || "https://via.placeholder.com/100"}
        alt={`${name}'s profile`}
      />

      {/* Name */}
      <h5 className="text-xl font-semibold text-white mb-1">{name}</h5>
      <span className="text-sm text-gray-400 mb-4">Developer</span>

      {/* Social Links */}
      <div className="flex gap-3 mt-2">
        {githubId && (
          <a
            href={`https://github.com/${githubId}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-white bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded-lg transition-all"
          >
            <svg
              className="w-4 h-4"
              fill="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path d="M12 .5C5.65.5.5 5.65.5 12A11.5 11.5 0 008.3 23.3c.6.1.8-.3.8-.6v-2.1c-3.3.7-4-1.6-4-1.6a3.2 3.2 0 00-1.3-1.7c-1-.7.1-.7.1-.7a2.6 2.6 0 011.9 1.3 2.7 2.7 0 003.7 1.1c.1-.8.4-1.5.8-1.9-2.7-.3-5.5-1.4-5.5-6a4.6 4.6 0 011.2-3.2 4.2 4.2 0 01.1-3.1s1-.3 3.3 1.2a11.4 11.4 0 016 0c2.3-1.5 3.3-1.2 3.3-1.2a4.2 4.2 0 01.1 3.1 4.6 4.6 0 011.2 3.2c0 4.6-2.8 5.7-5.5 6 .5.4.9 1.1.9 2.2v3.3c0 .3.2.7.8.6A11.5 11.5 0 0023.5 12C23.5 5.65 18.35.5 12 .5z" />
            </svg>
            GitHub
          </a>
        )}

        {linkedinId && (
          <a
            href={`https://linkedin.com/in/${linkedinId}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-white bg-[#0A66C2] hover:bg-[#004182] px-3 py-2 rounded-lg transition-all"
          >
            <svg
              className="w-4 h-4"
              fill="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path d="M4.98 3.5C4.98 5 3.92 6 2.5 6S0 5 0 3.5 1.08 1 2.5 1 4.98 2 4.98 3.5zM0 8h5v16H0V8zm7.5 0h4.7v2.2h.1a5.1 5.1 0 014.6-2.5c4.9 0 5.8 3.2 5.8 7.4V24h-5v-7.2c0-1.7 0-3.9-2.3-3.9s-2.7 1.8-2.7 3.8V24h-5V8z" />
            </svg>
            LinkedIn
          </a>
        )}
      </div>
    </div>
  );
};

export default Card;
