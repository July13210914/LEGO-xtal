# LEGO-Xtal: Local Environment Geometry-Oriented Crystal Generator

LEGO-Xtal (Local Environment Geometry-Oriented Crystal Generator) is designed for the rapid generation of crystal structures for complex materials characterized by well-defined local structural building units (SBUs). By leveraging a small set of known material examples, LEGO-Xtal learns target crystal structures by exploiting crystal symmetry. A generative model navigates the vast combinatorial space of symmetry operations to produce numerous trial structures, which are then optimized to ensure each atom satisfies the desired local atomic environment.

The LEGO-Xtal workflow is illustrated below using sp²-hybridized carbon structures as an example:
![LEGO-Xtal Framework](https://github.com/MaterSim/LEGO-xtal/blob/main/misc/Fig-framework.png)

## Workflow

1.  **Training Data Collection and Augmentation:**
    *   **Data Selection:** Curate a dataset of known structures (e.g., 140 sp² carbon structures from the SACADA database).
    *   **Crystal Representation:** Describe each structure using an irreducible crystal representation, encoding space group number, cell parameters, and Wyckoff site information (typically resulting in a ~40-column feature vector).
    *   **Symmetry-Based Data Augmentation:** Generate alternative representations based on group-subgroup relationships to maximize the utility of crystal symmetry. This allows the model to learn from diverse symmetry perspectives and ensures sufficient data for training.

2.  **Generative Model Training:**
    *   In the augmented tabular dataset (e.g., ~60,000 rows × 40 columns), transform each row to an extended feature representation using techniques like one-hot encoding and Gaussian mixture models to enhance model learning.
    *   Use the transformed feature table to train various generative models, such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), or Transformers.

3.  **Pre-relaxation:**
    *   Employ the trained models to generate new feature representations and then decode them to novel crystal structure candidates.
    *   Optimize each generated structure using L-BFGS or Adam optimizers to match a reference local environment, described by radial and Fourier distribution functions.

4.  **Energy Ranking and Database:**
    *   Rank the pre-relaxed structures with different energy models (e.g., ReaxFF, MACE or DFT)
    *   Compile the generated and validated structures into a database for further analysis.

By utilizing different local environments as training sources, LEGO-Xtal can rapidly generate high-quality crystal candidates for designing complex materials, including metal-organic frameworks (MOFs) and battery materials. Combining materials science domain knowledge with advanced AI methodologies, LEGO-Xtal aims to advance AI-driven crystal structure generation, paving the way for more efficient materials discovery and design.

---

## Installation

### 1. Install Julia in base
Ensure you have a clean installation of Julia. If you have a broken installation, remove `~/.juliaup` and `~/.julia` first.
```bash
rm -rf ~/.juliaup
rm -rf ~/.julia   
```
To install Juliaup (Still in base):
```bash
curl -fsSL https://install.julialang.org | sh
# Follow the prompts: install for current user -> yes, default location -> yes
```
After installation, reload your shell:
```bash
source ~/.bashrc
julia --version
```

### 2. Set up Conda Environment
Create a new environment and install Python dependencies.
```bash
conda create -n xtal python=3.10.8
conda activate xtal

# Install Python packages
pip install --upgrade git+https://github.com/MaterSim/PyXtal.git@master
pip install mace-torch==0.3.4
pip install torch-dftd==0.4.0
pip install gulp==0.1.0
```

### 3. Clean Conflicting Julia Installations
Ensure no conflicting Julia installations exist within the Conda environment.
```bash
conda activate xtal
conda remove julia
rm -rf $CONDA_PREFIX/julia_env
rm -rf $CONDA_PREFIX/share/julia
rm -f  $CONDA_PREFIX/bin/julia
```

### 4. Install Julia Packages
Install `CrystalNets` and `JSON` in the Julia environment.
```bash
julia -e 'import Pkg; Pkg.add(name="CrystalNets", version="0.4.9"); Pkg.add("JSON")'
```

### 5. Verification
To verify that `CrystalNets` is working correctly, first check the Julia script:

```bash
cd util/Julia_Installation/
julia process_topology.jl C.cif
```
*Expected output:* `[{"name":"dia","dim":3,"count":1}]`

Then run the Python test script:

```bash
python test_calculate_topology.py
```
*Expected output:*
```text
Batch output length: 11, expected: 11, elapsed: 13.74s
Row 7: 141 ['32i', '32i', '32i'] dim=3 aaa
Row 8: 155 ['18f', '18f', '18f'] dim=3 aaa
...
```

---

## Usage

### 1. Training
To train the model using the sample data:
```bash
python 1_train_sample.py --data data/train/train-v4.csv --epochs 250 --model VAE --sample 1000
```

**Using Slurm:**
```bash
# Edit util/slurm/run_train.sh to adjust epochs and sample count if needed
sbatch -J train-v4 util/slurm/run_train.sh
```

### 2. Relaxation
To relax the generated structures:
```bash
python 2_relax.py --ncpu 32 --csv data/sample/test.csv --end 100000
```

**Using Slurm:**
```bash
# This script runs relaxation and energy ranking
sbatch -J test util/slurm/run_relax.sh
```