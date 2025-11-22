import { Link } from "react-router-dom";

const DashboardCard = ({ title, description, to }) => {
  const cardContent = (
    <div className="bg-gray-600 p-4 rounded-2xl shadow-md hover:shadow-lg transition duration-300 cursor-pointer">
      <h4 className="text-xl font-semibold text-white mb-2">{title}</h4>
      <p className="text-white">{description}</p>
    </div>
  );
  return to ? (
    <Link to={to} tabIndex={0} style={{ textDecoration: "none" }}>
      {cardContent}
    </Link>
  ) : (
    cardContent
  );
};

export default DashboardCard;
