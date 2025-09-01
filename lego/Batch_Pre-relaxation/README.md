# Crystal Structure Batch Optimization in GPU with PyTorch Autograd (Adam)

The pipeline enables you to optimize crystal structure parameters using a descriptor-based loss in PyTorch, with fully batched, end-to-end differentiable computation enabling autograd backpropagation.

## Requirements

- CUDA-compatible GPU recommended

Required data files:
- `wyckoff_list.csv`: Wyckoff position data
- `test_data.csv`: Input crystal structure data

## Usage

### Basic Usage

Run the batch optimization with default settings:

```bash
python batch_opt.py
```

### Configuration

Customize parameters by editing the `Config` class in the script:

```python
@dataclass
class Config:
    csv_path: Path = Path("your_data.csv")        # Input CSV file
    batch_size: int = 250                         # Batch size
    results_dir: Path = Path("Output_B-250")      # Output directory
    lr: float = 2e-3                              # Learning rate
    num_steps: int = 250                          # Optimization steps
    device: torch.device = torch.device("cuda")   # Device
```

### Input Data Format

The input CSV should contain columns for:
- Space group number
- Lattice parameters (a, b, c, α, β, γ)
- Wyckoff positions and atomic coordinates

Example:

```csv
spg,a,b,c,alpha,beta,gamma,wp0,x0,y0,z0,wp1,x1,y1,z1,...
194,2.46,2.46,6.70,1.5708,1.5708,2.0944,9,0.333,0.667,0.25,...
```

## Output

### Directory Structure

```
Output_B-{batch_size}/
├── mof.db                    # SQLite database
├── final.db                  # Final unique structures
├── cifs/                     # CIF files
├── gulp_0/                   # Energy calculation files
└── out_{batch_size}.log      # Logs
```

### Generated Files

- **CIF Files**: Optimized crystal structures
- **Database**: Structure metadata and properties
- **Logs**: Optimization progress and errors

## Advanced Usage

### Custom Optimization Parameters

Adjust optimizer and scheduler settings:

```python
optimizer = torch.optim.AdamW([rep_batch], lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
)
```

### Validation Criteria

Set structure validation rules:

```python
criteria = {"CN": {"C": [3]}, "cutoff": 2.1}
```

### Descriptor Calculator Settings

Configure SO3 descriptor parameters:

```python
f0 = SO3(lmax=4, nmax=2, alpha=1.5, rcut=2.1, max_N=100)
```

## Monitoring and Debugging

### Real-time Monitoring

Track progress:

```bash
tail -f out_{batch_size}.log
```

### Performance Optimization

- **GPU Memory**: Lower `batch_size` if CUDA out-of-memory occurs
- **Speed**: Increase `batch_size` on high-memory GPUs
- **Convergence**: Tune `num_steps` and learning rate






