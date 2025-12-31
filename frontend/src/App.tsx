import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from './contexts/AuthContext';

// Pages
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Register from './pages/Register';
import DatasetsPage from './pages/DatasetsPage';
import NewDatasetPage from './pages/NewDatasetPage';
import DatasetDetailPage from './pages/DatasetDetailPage';
import ModelsPage from './pages/ModelsPage';
import NewModelPage from './pages/NewModelPage';
import ModelDetailPage from './pages/ModelDetailPage';
import TrainingPage from './pages/TrainingPage';
import NewTrainingPage from './pages/NewTrainingPage';
import TrainingDetailPage from './pages/TrainingDetailPage';
import SettingsPage from './pages/SettingsPage';

// Layout
import Layout from './components/layout/Layout';

// Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});

// Protected route wrapper
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

// Public route wrapper (redirect if already logged in)
function PublicRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}

function AppRoutes() {
  return (
    <Routes>
      {/* Public routes */}
      <Route
        path="/login"
        element={
          <PublicRoute>
            <Login />
          </PublicRoute>
        }
      />
      <Route
        path="/register"
        element={
          <PublicRoute>
            <Register />
          </PublicRoute>
        }
      />

      {/* Protected routes */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Dashboard />} />

        {/* Dataset routes */}
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="datasets/new" element={<NewDatasetPage />} />
        <Route path="datasets/:id" element={<DatasetDetailPage />} />

        {/* Model routes */}
        <Route path="models" element={<ModelsPage />} />
        <Route path="models/new" element={<NewModelPage />} />
        <Route path="models/:id" element={<ModelDetailPage />} />

        {/* Training routes */}
        <Route path="training" element={<TrainingPage />} />
        <Route path="training/new" element={<NewTrainingPage />} />
        <Route path="training/:id" element={<TrainingDetailPage />} />

        {/* Settings */}
        <Route path="settings" element={<SettingsPage />} />
      </Route>

      {/* Catch all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
