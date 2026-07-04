import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Edges } from '@react-three/drei';
import * as THREE from 'three';

interface ThreeVisualizerProps {
  voxelShape: [number, number, number];
}

export const ThreeVisualizer: React.FC<ThreeVisualizerProps> = ({ voxelShape }) => {
  // Calculate the center to correctly point the camera and orbit controls
  const center = useMemo(() => {
    if (!voxelShape || voxelShape.length !== 3) return new THREE.Vector3(10, 10, 10);
    return new THREE.Vector3(voxelShape[0] / 2, voxelShape[1] / 2, voxelShape[2] / 2);
  }, [voxelShape]);

  return (
    <Canvas camera={{ position: [center.x * 2, center.y * 2.5, center.z * 2.5], fov: 50 }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[20, 30, 20]} intensity={1.5} />
      
      {/* Allows the user to drag and rotate the 3D model */}
      <OrbitControls target={center} makeDefault />
      
      {/* 
        MVP Visualization: Renders the bounding box of the generated Voxel Matrix.
        In the full implementation, we will map over the binary voxel matrix 
        and render individual <boxGeometry args={[1,1,1]}> for every bead.
      */}
      <mesh position={center}>
        <boxGeometry args={voxelShape || [20, 20, 20]} />
        <meshStandardMaterial color="#8b5cf6" transparent opacity={0.4} roughness={0.1} metalness={0.5} />
        <Edges scale={1.0} threshold={15} color="#ffffff" />
      </mesh>
      
      {/* Pegboard Ground Plane visualization */}
      <gridHelper args={[40, 40, '#4f46e5', '#334155']} position={[center.x, 0, center.z]} />
    </Canvas>
  );
};
