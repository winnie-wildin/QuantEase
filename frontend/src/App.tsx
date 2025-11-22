import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import { Home } from './pages/Home';
import { CreateExperiment } from './pages/CreateExperiment';
import { UploadDataset } from './pages/UploadDataset';
import { SetupModels } from './pages/SetupModels';
import { ExperimentDetails } from './pages/ExperimentDetails';
import { SampleComparison } from './pages/SampleComparison';
import { Toaster } from 'react-hot-toast';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#10b981',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/create" element={<CreateExperiment />} />
              <Route path="/experiment/:id/upload" element={<UploadDataset />} />
              <Route path="/experiment/:id/setup" element={<SetupModels />} />
              <Route path="/experiment/:id" element={<ExperimentDetails />} />
              <Route path="/experiment/:id/samples" element={<SampleComparison />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </>
    </QueryClientProvider>
  );
}

export default App;