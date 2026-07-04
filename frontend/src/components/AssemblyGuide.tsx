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

  if (!instructions || instructions.length === 0) {
    return <p>No assembly instructions generated.</p>;
  }

  const step = instructions[currentStep];

  return (
    <div style={{ marginTop: '1rem' }}>
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
        
        {/* Render a 2D mini pegboard grid */}
        <div style={{ 
          display: 'grid', 
          gap: '2px', 
          gridTemplateColumns: `repeat(${step.layout[0]?.length || 20}, 1fr)`,
          background: 'rgba(0,0,0,0.3)',
          padding: '10px',
          borderRadius: '8px'
        }}>
          {step.layout.map((row, y) => 
            row.map((cell, x) => (
              <div 
                key={`${x}-${y}`} 
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
        <p style={{ fontSize: '0.8rem', opacity: 0.6, marginTop: '1rem', textAlign: 'center' }}>
          Place beads on your pegboard matching the purple pattern above.
        </p>
      </div>
    </div>
  );
};
