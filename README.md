# Attention-Based Face Inpainting using GANs

This project explores a **context-aware GAN architecture** for realistic face inpainting, capable of reconstructing occluded facial regions using attention mechanisms, segmentation masks, and feature conditioning.

---

## 🧠 Project Highlights
- GAN with **contextual attention** for facial region inpainting  
- **U-Net segmentation** to detect occluded areas  
- **EfficientNet-based classifier** for feature extraction  
- Supports **feature-guided inpainting** using embedding vectors  
- Evaluated with **PSNR, SSIM, IoU, and Dice Score**

---

## 📂 Dataset
- **Source**: CelebA  
- **Preprocessing**:
  - Detected faces using **Haar Cascade**
  - Generated 6,000 **occluded images** with 20 types of occlusions
  - Created binary (6000) and inverted **masks** (6000) for each occlusion
  - Resulting in total ~18000 images for training
---

## 🧪 Model Components

### 1. Segmentation Module
- Architecture: **U-Net**  
- Output: Binary occlusion mask  
- Metrics: **IoU**, **Dice Score**

### 2. Feature Classifier
- Architecture: Modified **EfficientNet-B0**  
- Accuracy: **99.11%** on occluded face identity classification  
- Output: Feature vector for generator conditioning

### 3. Stable Generator (GAN)
- Inputs: Occluded image, binary mask, feature vector  
- Architecture:
  - Stable gated convolutions
  - Contextual attention mechanism
  - Deep skip-connected decoder  
- Output: Reconstructed facial image

### 4. Discriminator
- Objective: Distinguish inpainted vs. real images  
- Integrated with generator for adversarial training

---

## 📊 Evaluation Metrics
- Classification Accuracy on occluded faces 99.11%
- **IoU** > 0.8
- **Dice** > 0.8
- **PSNR** > 38  
- **SSIM** > 0.98  
- Segmentation quality: High **IoU** and **Dice Score**

---

## ⚙️ Tech Stack
- Python, **PyTorch**, OpenCV  
- DeepFace (for identity embedding comparisons)  
- NumPy, Matplotlib

---

## 📎 Results
- Successfully inpainted occluded faces with high realism  
- Maintained identity using classifier guidance  
- Outperformed traditional GAN baselines in visual and metric-based comparison
![image](https://github.com/user-attachments/assets/2e1fe98c-ed83-433b-ba1f-4c389e4b2d9f)

---

## 🧾 License
MIT License

---

## 👨‍💻 Author
**[Varshil Prajapati]** – MTech AI | Researcher | GANs & Vision Enthusiast  
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/varshil-prajapati-610251223/) or explore more projects!
