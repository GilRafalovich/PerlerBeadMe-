import os
import json
import urllib.request
import zipfile
import numpy as np
import trimesh
import cv2
from tqdm import tqdm
from scipy.spatial import cKDTree
import argparse

from quantizer import VoxelQuantizer
from voxelizer import VertexAI3DEstimator

import kagglehub

def generate_ground_truth_voxel_matrix(obj_path, max_dim=60):
    try:
        mesh = trimesh.load(obj_path, force='mesh')
        max_extent = np.max(mesh.extents)
        if max_extent == 0: return None
        voxel_grid = mesh.voxelized(pitch=max_extent/max_dim).fill()
        m = voxel_grid.matrix
        padded = np.zeros((max_dim, max_dim, max_dim), dtype=bool)
        x, y, z = min(max_dim, m.shape[0]), min(max_dim, m.shape[1]), min(max_dim, m.shape[2])
        padded[:x, :y, :z] = m[:x, :y, :z]
        return padded
    except Exception as e:
        return None

def calculate_iou(mat1, mat2):
    intersection = np.logical_and(mat1, mat2).sum()
    union = np.logical_or(mat1, mat2).sum()
    return intersection / union if union > 0 else 0

def calculate_chamfer_distance(mat1, mat2):
    pts1 = np.argwhere(mat1)
    pts2 = np.argwhere(mat2)
    if len(pts1) == 0 or len(pts2) == 0: return 999.0
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    d1, _ = tree1.query(pts2)
    d2, _ = tree2.query(pts1)
    return np.mean(d1) + np.mean(d2)

def run_verification(limit=100):
    print("--- Pix3D End-to-End Verification Pipeline ---")
    print("Downloading Pix3D via KaggleHub (This might take a minute)...")
    try:
        data_dir = kagglehub.dataset_download("divyansh6349/pix3d-dataset")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return
        
    print(f"Pix3D Data successfully loaded from {data_dir}!")
    
    # Locate the pix3d.json file
    json_path = os.path.join(data_dir, 'pix3d.json')
    if not os.path.exists(json_path):
        # Sometimes kaggle datasets have nested folders
        json_path = os.path.join(data_dir, 'pix3d', 'pix3d.json')
        data_dir = os.path.join(data_dir, 'pix3d')
        if not os.path.exists(json_path):
            print("ERROR: Could not find pix3d.json in the downloaded dataset.")
            return
            
    print("Loading Pix3D Annotations...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if limit > 0:
        import random
        random.shuffle(data)
        data = data[:limit]
        print(f"Randomly selected {limit} samples for testing.")
    else:
        print(f"Running full dataset evaluation on {len(data)} samples!")
        
    quantizer = VoxelQuantizer(max_dim_xy=60)
    ai_estimator = VertexAI3DEstimator()
    
    ious = []
    cds = []
    
    results = []
    
    for item in tqdm(data, desc="Processing Pix3D"):
        img_path = os.path.join(data_dir, item['img'])
        gt_obj_path = os.path.join(data_dir, item['model'])
        
        if not os.path.exists(img_path) or not os.path.exists(gt_obj_path):
            continue
            
        # 1. Generate Ground Truth Matrix
        gt_matrix = generate_ground_truth_voxel_matrix(gt_obj_path)
        if gt_matrix is None: continue
        
        # 2. Run AI Pipeline (Real 2D Image -> AI Generated Mesh -> Kid-Friendly Voxel Matrix)
        generated_obj_path = "tmp_generated.obj"
        try:
            ai_estimator.generate_3d_mesh(img_path, generated_obj_path)
            cv2_img = cv2.imread(img_path)
            _, _, kid_matrix = quantizer.spatial_quantize(cv2_img, generated_obj_path)
        except Exception as e:
            continue
            
        # 3. Calculate Metrics
        iou = calculate_iou(kid_matrix, gt_matrix)
        cd = calculate_chamfer_distance(kid_matrix, gt_matrix)
        
        # 4. LLM Verification
        from llm_evaluator import evaluate_with_gemini
        llm_score, llm_critique = evaluate_with_gemini(img_path, kid_matrix)
        
        ious.append(iou)
        cds.append(cd)
        
        results.append({
            "image": item['img'],
            "category": item.get('category', 'unknown'),
            "iou": iou,
            "chamfer_distance": cd,
            "llm_score": llm_score,
            "llm_critique": llm_critique
        })
        
    if len(ious) > 0:
        avg_iou = float(np.mean(ious) * 100)
        avg_cd = float(np.mean(cds))
        print("\n=== FINAL METRICS ===")
        print(f"Evaluated {len(ious)} models End-to-End.")
        print(f"Average 3D IoU: {avg_iou:.2f}% (Expect low due to intentional hollowing)")
        print(f"Average Chamfer Distance: {avg_cd:.4f} (Lower is better for structural match)")
        
        # Save to JSON
        os.makedirs("test_results", exist_ok=True)
        out_path = os.path.join("test_results", "verification_results.json")
        output_data = {
            "summary": {
                "total_evaluated": len(ious),
                "average_iou_percent": avg_iou,
                "average_chamfer_distance": avg_cd
            },
            "runs": results
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results saved to {out_path}")
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of random samples to run. Set to -1 for full dataset.")
    args = parser.parse_args()
    run_verification(limit=args.limit)
