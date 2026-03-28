import os
import safetensors.torch
import comfy.utils
import hashlib
import folder_paths

class CustomLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "temp.latent"})
            }
        }

    CATEGORY = "Custom"

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    OUTPUT_NODE = True

    def load(self, file_path):
        if not os.path.isabs(file_path):
            file_path = os.path.join(folder_paths.get_input_directory(), file_path)
        # Ensure the file exists and is the latest version
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        # Reload the file to ensure it's up-to-date
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Load the latent file
        latent = safetensors.torch.load_file(file_path, device="cpu")
        multiplier = 1.0
        if "latent_format_version_0" not in latent:
            multiplier = 1.0 / 0.18215
        samples = {"samples": latent["latent_tensor"].float() * multiplier}
        return (samples,)

    @classmethod
    def IS_CHANGED(cls, file_path):
        if not os.path.isabs(file_path):
            file_path = os.path.join(folder_paths.get_input_directory(), file_path)
        image_path = file_path
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, file_path):
        if not os.path.isabs(file_path):
            file_path = os.path.join(folder_paths.get_input_directory(), file_path)
        if not os.path.exists(file_path):
            return "Invalid latent file: {}".format(file_path)
        return True

# Define NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "CustomLoadLatent": CustomLoadLatent
}
