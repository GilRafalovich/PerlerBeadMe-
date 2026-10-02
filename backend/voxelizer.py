import os
import requests


class HuggingFace3DEstimator:
    """
    Communicates with a TripoSR HTTP endpoint (default: HuggingFace Space)
    to completely automate the 3D generation process.
    """

    def __init__(self):
        self.endpoint = os.environ.get(
            "REMOTE_3D_ENDPOINT_URL",
            "https://ahmedbelaid1-triposr.hf.space/generate",
        )
        print(f"Using TripoSR HTTP endpoint: {self.endpoint}")

    def generate_3d_mesh(self, image_path: str, output_obj_path: str):
        """
        Sends the 2D image to TripoSR via HTTP POST (background removal + mesh),
        and writes the returned .obj bytes to output_obj_path.
        """
        print(f"1. POSTing image to TripoSR: {image_path}")
        with open(image_path, "rb") as image_file:
            files = {
                "image": (os.path.basename(image_path), image_file, "image/png"),
            }
            data = {
                "do_remove_background": "true",
                "foreground_ratio": "0.85",
                "mc_resolution": "256",
                "format": "obj",
                # Default bake_texture_flag=true returns a ZIP; false returns raw mesh bytes.
                "bake_texture_flag": "false",
            }
            response = requests.post(
                self.endpoint,
                files=files,
                data=data,
                timeout=1200,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"TripoSR request failed: HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        print("2. Writing 3D mesh...")
        os.makedirs(os.path.dirname(os.path.abspath(output_obj_path)) or ".", exist_ok=True)
        with open(output_obj_path, "wb") as out:
            out.write(response.content)
        print(f"Successfully retrieved 3D mesh: {output_obj_path} ({len(response.content)} bytes)")
        return output_obj_path
