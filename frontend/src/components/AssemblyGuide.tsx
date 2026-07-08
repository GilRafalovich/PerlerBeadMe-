import React, { useState } from 'react';

interface Instruction {
  type: string;
  z_index?: number;
  x_index?: number;
  layout: number[][];
  dowel_holes?: any[];
}

interface AssemblyGuideProps {
  instructions: Instruction[];
}

export const AssemblyGuide: React.FC<AssemblyGuideProps> = ({ instructions }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [viewMode, setViewMode] = useState<'step' | 'debug'>('step');

  if (!instructions || instructions.length === 0) {
    return <p>No assembly instructions generated.</p>;
  }

  const step = instructions[currentStep];

  const renderGrid = (layout: number[][], keyPrefix: string) => (
    <div style={{ 
      display: 'grid', 
      gap: '2px', 
      gridTemplateColumns: `repeat(${layout[0]?.length || 20}, 1fr)`,
      background: 'rgba(0,0,0,0.3)',
      padding: '10px',
      borderRadius: '8px'
    }}>
      {layout.map((row, y) => 
        row.map((cell, x) => (
          <div 
            key={`${keyPrefix}-${x}-${y}`} 
            style={{ 
              aspectRatio: '1/1',
              background: cell ? '#8b5cf6' : 'transparent',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '50%'
            }} 
          />
        ))
      )}
    </div>
  );

  return (
    <div style={{ marginTop: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem', gap: '1rem' }}>
        <button 
          className="btn" 
          onClick={() => setViewMode('step')}
          style={{ opacity: viewMode === 'step' ? 1 : 0.5, padding: '6px 12px', fontSize: '0.9rem' }}
        >
          Step-by-Step
        </button>
        <button 
          className="btn" 
          onClick={() => setViewMode('debug')}
          style={{ opacity: viewMode === 'debug' ? 1 : 0.5, padding: '6px 12px', fontSize: '0.9rem' }}
        >
          Debug Tile View
        </button>
      </div>

      {viewMode === 'step' ? (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <button 
              className="btn" 
              onClick={() => setCurrentStep(prev => Math.max(0, prev - 1))}
              disabled={currentStep === 0}
              style={{ padding: '8px 16px', fontSize: '0.9rem' }}
            >
              &larr; Prev
            </button>
            <span style={{ alignSelf: 'center', fontWeight: 'bold' }}>
              Step {currentStep + 1} of {instructions.length}
            </span>
            <button 
              className="btn" 
              onClick={() => setCurrentStep(prev => Math.min(instructions.length - 1, prev + 1))}
              disabled={currentStep === instructions.length - 1}
              style={{ padding: '8px 16px', fontSize: '0.9rem' }}
            >
              Next &rarr;
            </button>
          </div>

          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '8px' }}>
            <h3 style={{ margin: '0 0 1rem 0', color: '#e2e8f0' }}>
              {step.type === 'horizontal_plane' ? `Build Horizontal Z-Layer ${step.z_index}` : `Build Vertical X-Wall ${step.x_index}`}
            </h3>
            
            {renderGrid(step.layout, 'step')}
            <p style={{ fontSize: '0.8rem', opacity: 0.6, marginTop: '1rem', textAlign: 'center' }}>
              Place beads on your pegboard matching the purple pattern above.
            </p>
          </div>
        </>
      ) : (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', 
          gap: '1rem',
          maxHeight: '400px',
          overflowY: 'auto',
          padding: '10px'
        }}>
          {instructions.map((inst, idx) => (
            <div key={idx} style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem', borderRadius: '8px' }}>
              <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.8rem', color: '#e2e8f0', textAlign: 'center' }}>
                {inst.type === 'horizontal_plane' ? `Z-Layer ${inst.z_index}` : `X-Wall ${inst.x_index}`}
              </h4>
              {renderGrid(inst.layout, `debug-${idx}`)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
