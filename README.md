# HiC-GNN-for-GPU: A Generalizable Model for 3D Chromosome Reconstruction Using Graph Convolutional Neural Networks (With GPU)
------------------------------------------------------------------------------------------------------------------------------------
Refactored from the original academic code, this version is designed for **universal compatibility**, running seamlessly on everything from standard CPUs to bleeding-edge GPUs (e.g., NVIDIA Blackwell, H100, RTX 6000 Ada) without dependency conflicts. It replaces deprecated libraries (`LINE`,`ge`, `tensorflow 1.x`) with modern **PyTorch** and **Node2Vec**. Replaced external R scripts with a native Python implementation of Knight-Ruiz (KR) normalization, eliminating complex environment setups

## 📂 Repository Structure

```text
HiC-GNN-for-GPU/
├── data/                 # Input Hi-C contact maps (e.g., GM12878_1mb_chr19_list.txt)
├── src/                  # Core source code
│   ├── main.py           # Main training and prediction script
│   ├── models.py         # Pure PyTorch GNN model definitions
│   ├── layers.py         # Custom GraphSAGE layers (Kernel-free)
│   ├── embeddings.py     # Node2Vec structural embedding logic
│   ├── normalization.py  # Python implementation of Knight-Ruiz (KR) normalization
│   └── utils.py          # Data loading, PDB writing, and seeding tools
├── Outputs/              # Generated results (PDB structures, model weights, logs)
├── environment.yml       # Conda environment configuration
├── run_benchmark.py      # Script to test performance and speed
└── README.md             # Project documentation
```

## 🛠️ Installation

We recommend using **Conda** to manage dependencies.

### 1. Clone the Repository
```bash
git clone https://github.com/fahimulkabir/HiC-GNN-for-GPU.git
cd HiC-GNN-for-GPU
```

### 2. Create the Environment
Create the `environment.yml` file with the following content (or use the file provided in the repo):

**`environment.yml`**
```yaml
name: hicgnn-gpu
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - networkx
  - scikit-learn
  - pip
  - pip:
    - node2vec
```

Run the creation command:
```bash
conda env create -f environment.yml
conda activate hicgnn-gpu
```

### 3. Install PyTorch (Hardware Specific)
Since drivers vary between servers (e.g., H100 vs. RTX 4090), install the version of PyTorch that matches your hardware:

**Option A: Standard GPUs (H100, A100, RTX 3090/4090)**
*For systems with CUDA 11.8 - 12.x drivers.*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Option B: Bleeding Edge GPUs (Blackwell / RTX 6000 Ada)**
*For new systems requiring CUDA 13.0+.*
```bash
pip install torch torchvision --pre --extra-index-url https://download.pytorch.org/whl/cu130
```

**Option C: CPU Only**
*For laptops or non-GPU servers.*
```bash
pip install torch torchvision
```

## 🚀 Usage

### Training & Prediction
To generate a 3D structure from a contact map, run the main module. The script automatically handles normalization, embedding generation, and training.

```bash
# Syntax: python -m src.main <path_to_data_file>
python -m src.main data/GM12878_1mb_chr19_list.txt
```

### Benchmarking
To test the speed and accuracy of your setup across multiple runs:

```bash
python run_benchmark.py
```

## 📊 Output Files
After a successful run, the following files are saved in the `Outputs/` directory:

* **`*_structure.pdb`**: The predicted 3D genome structure. Capable of being visualized in PyMOL, UCSF Chimera, or other molecular viewers.
* **`*_weights.pt`**: The trained PyTorch model weights.
* **Logs**: Training logs containing the Loss and Distance Spearman Correlation Coefficient (dSCC) for the optimal Alpha parameter.

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
