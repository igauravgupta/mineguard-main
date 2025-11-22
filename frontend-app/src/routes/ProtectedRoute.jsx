import { useAuth, RedirectToSignIn } from "@clerk/clerk-react";

export default function ProtectedRoute({ children }){
  const { isLoaded, isSignedIn } = useAuth();

  if (!isLoaded) return <div>Loading...</div>;
  if (!isSignedIn) return <RedirectToSignIn />;

  return children;
}
