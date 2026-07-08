from datasets import load_dataset

def test_hf():
    print("Loading HF dataset in streaming mode...")
    try:
        dataset = load_dataset("ArkAiLab-Adl/Nexora-3d-mesh-dataset-v1-mini", split="train", streaming=True)
        sample = next(iter(dataset))
        print("Keys:", sample.keys())
        print("Successfully loaded.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_hf()
