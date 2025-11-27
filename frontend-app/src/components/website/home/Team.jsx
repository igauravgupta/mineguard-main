// // src/pages/AboutPage.js
// import Card from "./Cards.jsx";

// const TeamPage = () => {
//   // Developer data
//   const team = [
//     {
//       name: "Gaurav Sharma",
//       githubId: "gauravsharma",
//       linkedinId: "gaurav-sharma",
//       image: "https://avatars.githubusercontent.com/u/9919?v=4",
//     },
//     {
//       name: "Aarav Mehta",
//       githubId: "aaravmehta",
//       linkedinId: "aarav-mehta",
//       image: "https://avatars.githubusercontent.com/u/583231?v=4",
//     },
//     {
//       name: "Riya Patel",
//       githubId: "riyapatel",
//       linkedinId: "riya-patel",
//       image: "https://avatars.githubusercontent.com/u/810438?v=4",
//     },
//   ];

//   return (
//     <div className="bg-gray-900 text-white">
//       {/* Meet Developers Section */}
//       <section className="py-20 px-6">
//         <div className="max-w-6xl mx-auto text-center">
//           {/* Decorative line */}
//           <div className="w-16 h-1 bg-purple-500 rounded-full mx-auto mb-6" />

//           {/* Section heading */}
//           <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
//             Meet Our Developers
//           </h2>

//           <p className="text-gray-400 text-base sm:text-lg max-w-2xl mx-auto mb-12">
//             Passionate minds who bring ideas to life with creativity, skill, and
//             collaboration.
//           </p>

//           {/* Developer cards */}
//           <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3 justify-items-center">
//             {team.map((member) => (
//               <Card
//                 key={member.githubId}
//                 name={member.name}
//                 githubId={member.githubId}
//                 linkedinId={member.linkedinId}
//                 image={member.image}
//               />
//             ))}
//           </div>
//         </div>
//       </section>
//     </div>
//   );
// };

// export default TeamPage;
