# Fine-Tuning LLM with LoRA

This project provides a comprehensive toolkit for fine-tuning large language models (LLMs) using Low-Rank Adaptation (LoRA). It includes scripts for data preparation, model training, evaluation, and comparison.

## Features

- Data preparation and augmentation
- LoRA-based model fine-tuning
- Training monitoring and visualization
- Model evaluation and comparison
- System testing and validation

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

## Hardware Requirements

For fine-tuning the Falcon-7B model, the following GPU specifications are recommended:

### Minimum Requirements

- NVIDIA GPU with at least 16GB VRAM
- Examples: RTX 3080 (10GB), RTX 3080 Ti (12GB), RTX 4080 (16GB)

### Recommended for Better Performance

- NVIDIA GPU with 24GB+ VRAM
- Examples:
  - RTX 4090 (24GB)
  - A5000 (24GB)
  - A6000 (48GB)

### Best Performance

- NVIDIA A100 (40GB or 80GB)
- H100 (80GB)

### Cloud Provider Options

- AWS: p3.2xlarge (16GB) or p3.8xlarge (32GB)
- Google Cloud: A2 instance with NVIDIA A100
- Azure: NC6s_v3 or ND96amsr_A100

Note: The actual VRAM requirements will depend on your batch size, sequence length, and whether you're using mixed precision training. With LoRA, the VRAM requirements are significantly reduced compared to full model fine-tuning.

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd fine-tuning-llm
```

2. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `prepare_data.py`: Script for preparing and preprocessing training data
- `augment_data.py`: Data augmentation utilities
- `train_lora.py`: Main script for fine-tuning models using LoRA
- `monitor_training.py`: Training progress monitoring and visualization
- `evaluate_model.py`: Model evaluation and metrics calculation
- `compare_models.py`: Script for comparing different model versions
- `lora_config.py`: Configuration settings for LoRA implementation
- `test_cases.json`: Test cases for model evaluation
- `run_pod/`: Directory containing pod-related configurations

## Usage

1. Prepare your data:

```bash
python prepare_data.py
```

2. Fine-tune the model:

```bash
python train_lora.py
```

3. Monitor training progress:

```bash
python monitor_training.py
```

4. Evaluate the model:

```bash
python evaluate_model.py
```

5. Compare different model versions:

```bash
python compare_models.py
```

## Configuration

The `lora_config.py` file contains various configuration options for the LoRA implementation, including:

- Model architecture settings
- Training parameters
- LoRA-specific configurations

## System Test

Checks hardware to determine if hardware is CUDA comaptable:

```bash
python system_test.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
