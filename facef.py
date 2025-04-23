import numpy as np
from deepface import DeepFace
import io
from PIL import Image
import os
class FeatureExtractor:
    def __init__(self, model_name="Facenet", enforce_detection=True):
        self.model_name = model_name
        self.enforce_detection = enforce_detection

    def extract_features(self, img_tensor):
        try:
            img_path = self.tensor_to_numpy_via_png(img_tensor)

            # Use DeepFace to extract features from the image (without saving it to disk)
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=self.enforce_detection
            )
            #print(f"Extracted features: {embedding_objs}")
            if embedding_objs:
                print(np.array(embedding_objs[0]["embedding"]).shape)
                return np.array(embedding_objs[0]["embedding"])
            else:
                return None
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
        
    def tensor_to_numpy_via_png(self, img_tensor, filename="temp_image.png"):
        # Step 1: Convert tensor to PIL image
        img = img_tensor.cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)

        # Step 2: Save as PNG
        pil_img.save(filename)

        # Step 3: Read image back as NumPy array
        reloaded_img = Image.open(filename).convert("RGB")
        #img_np = np.array(reloaded_img)

        return filename

