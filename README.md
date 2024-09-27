# MRI Reconstruction

## 1. Installation

Install the required packages using:

```
conda env create -f environment.yml
```


## 2. Repository Structure

```
.
├── README.md
├── environment.yml
├── multi_gpu
│   ├── config
│   │   ├── config.yaml
│   │   └── config_utils.py
│   ├── data_utils.py
│   ├── datasets.py
│   ├── main.py
│   ├── model.py
│   └── train_utils.py
├── multi_vol
│   ├── config
│   │   ├── config.yaml
│   │   └── config_utils.py
│   ├── data_utils.py
│   ├── datasets.py
│   ├── main.py
│   ├── model.py
│   └── train_utils.py
├── run_exp.sh
└── single_vol
    ├── config
    │   ├── config.yaml
    │   └── config_utils.py
    ├── data_utils.py
    ├── datasets.py
    ├── main.py
    ├── model.py
    └── train_utils.py
```

The project is divided into two main settings:

1. **Single-Volume Setting**: 
   - Code located in the `single_vol/` folder
   - Focuses on reconstructing a single MRI volume

2. **Multi-Volume Setting**:
   - `multi_vol/` folder: Runs on a single GPU
   - `multi_gpu/` folder: Runs on multiple GPUs


## 3. Execution

To run the code, use:

```
sbatch run_exp.sh
```

### Configuration

Before running, ensure you modify the following:

1. `run_exp.sh`:
	- Update `source path_to_conda/conda.sh` with the correct path to your conda installation
	- Specify the correct `main.py` to run (e.g., `multi_gpu/main.py`, `multi_vol/main.py`, or `single_vol/main.py`)
	- For multi-GPU execution, uncomment the lines defining `MASTER_PORT` and `MASTER_ADDR` environment variables

2. `config.yaml`:
	- `path_to_outputs`: Set the path where tensorboard outputs and checkpoints will be saved
	- `path_to_data`: 
		- For multi-volume: path to the dataset folder
		- For single-volume: path to the MRI volume file
	- (Optional) `model_checkpoint`: Path to a saved model for initialization
	- For multi-volume setting:
		- `runtype`: Set to "train" for training phase, "test" for inference phase
