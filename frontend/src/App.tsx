import React, { useState } from 'react';
import './App.css';
import { ThreeVisualizer } from './components/ThreeVisualizer';
import { AssemblyGuide } from './components/AssemblyGuide';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    setLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/process_image', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error processing image:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>PerlerBeadMe</h1>
        <p style={{ textAlign: 'center', opacity: 0.8 }}>
          Turn any image into an Assembly-Optimized 3D Perler Bead Model
        </p>
      </header>

      {!result ? (
        <div className="glass-panel">
          <div 
            className="dropzone" 
            onDragOver={(e) => e.preventDefault()} 
            onDrop={handleDrop}
          >
            {file ? (
              <div>
                <h3>{file.name}</h3>
                <button className="btn" onClick={handleProcess} style={{ marginTop: '1rem' }}>
                  {loading ? 'Processing...' : 'Generate 3D Model'}
                </button>
              </div>
            ) : (
              <h3>Drag & Drop an image here</h3>
            )}
          </div>
        </div>
      ) : (
        <div className="viewer-grid">
          <div className="glass-panel">
            <h2>3D Preview</h2>
            <div style={{ height: '400px', background: 'rgba(0,0,0,0.5)', borderRadius: '8px', overflow: 'hidden' }}>
              <ThreeVisualizer voxelShape={result.voxel_shape} />
            </div>
          </div>
          <div className="glass-panel">
            <h2>Assembly Guide</h2>
            <AssemblyGuide instructions={result.instructions} />
            <button className="btn" onClick={() => setResult(null)} style={{ marginTop: '2rem', width: '100%' }}>
              Start Over With New Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
