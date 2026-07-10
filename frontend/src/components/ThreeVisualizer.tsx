import React, { useMemo, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Edges } from '@react-three/drei';
import * as THREE from 'three';

interface ThreeVisualizerProps {
  voxelShape: [number, number, number];
  voxelMatrix?: number[][][];
  colorMatrix?: number[][][];
}

export const ThreeVisualizer: React.FC<ThreeVisualizerProps> = ({ voxelShape, voxelMatrix, colorMatrix }) => {
  // Calculate the center to correctly point the camera and orbit controls
  const center = useMemo(() => {
    if (!voxelShape || voxelShape.length !== 3) return new THREE.Vector3(10, 10, 10);
    return new THREE.Vector3(voxelShape[0] / 2, voxelShape[1] / 2, voxelShape[2] / 2);
  }, [voxelShape]);

  // Extract instances data
  const instances = useMemo(() => {
    const items: { position: [number, number, number]; color: THREE.Color }[] = [];
    if (voxelMatrix && colorMatrix) {
      for (let x = 0; x < voxelMatrix.length; x++) {
        for (let y = 0; y < voxelMatrix[x].length; y++) {
          for (let z = 0; z < voxelMatrix[x][y].length; z++) {
            if (voxelMatrix[x][y][z] === 1) {
              const colorArray = colorMatrix[y][x]; 
              const color = new THREE.Color(`rgb(${colorArray[0]}, ${colorArray[1]}, ${colorArray[2]})`);
              // Map: X -> X, Z (height in matrix) -> Y (height in 3D), Y -> Z
              items.push({ position: [x, z, y], color });
            }
          }
        }
      }
    }
    return items;
  }, [voxelMatrix, colorMatrix]);

  const meshRef = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    if (meshRef.current && instances.length > 0) {
      const dummy = new THREE.Object3D();
      instances.forEach((inst, i) => {
        dummy.position.set(...inst.position);
        dummy.updateMatrix();
        meshRef.current!.setMatrixAt(i, dummy.matrix);
        meshRef.current!.setColorAt(i, inst.color);
      });
      meshRef.current.instanceMatrix.needsUpdate = true;
      if (meshRef.current.instanceColor) {
        meshRef.current.instanceColor.needsUpdate = true;
      }
    }
  }, [instances]);

  const geometry = useMemo(() => new THREE.BoxGeometry(0.9, 0.9, 0.9), []);
  const material = useMemo(() => new THREE.MeshStandardMaterial({ roughness: 0.3, metalness: 0.1 }), []);

  return (
    <Canvas camera={{ position: [center.x * 2, center.y * 2.5, center.z * 2.5], fov: 50 }}>
      <ambientLight intensity={0.8} />
      <directionalLight position={[20, 30, 20]} intensity={1.5} />
      
      {/* Allows the user to drag and rotate the 3D model */}
      <OrbitControls target={center} makeDefault />
      
      {instances.length > 0 ? (
        <instancedMesh ref={meshRef} args={[geometry, material, instances.length]} />
      ) : (
        <mesh position={center}>
          <boxGeometry args={voxelShape || [20, 20, 20]} />
          <meshStandardMaterial color="#8b5cf6" transparent opacity={0.4} roughness={0.1} metalness={0.5} />
          <Edges scale={1.0} threshold={15} color="#ffffff" />
        </mesh>
      )}
      
      {/* Pegboard Ground Plane visualization */}
      <gridHelper args={[40, 40, '#4f46e5', '#334155']} position={[center.x, -0.5, center.z]} />
    </Canvas>
  );
};
