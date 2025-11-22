import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Plus } from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  
  const isActive = (path: string) => location.pathname === path;
  
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-2">
  <Link to="/" className="flex items-center space-x-3">
    <img 
      src="/quanteaselogoo_2.jpg" 
      alt="QuantEase"
      className="w-20 h-20"
    />
    <div>
  <h1 className="text-3xl font-bold text-gray-900">QuantEase</h1>
  <p className="text-sm text-gray-600 font-medium">LLM Quantization Comparison</p>
</div>
  </Link>

  <nav className="flex space-x-4">
    <Link
      to="/"
      className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${
        isActive('/') ? 'bg-primary-100 text-primary-700' : 'text-gray-600 hover:bg-gray-100'
      }`}
    >
      <Home className="w-4 h-4" />
      <span>Experiments</span>
    </Link>

    <Link
      to="/create"
      className="btn-primary flex items-center space-x-2"
    >
      <Plus className="w-4 h-4" />
      <span>New Experiment</span>
    </Link>
  </nav>
</div>

        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
};