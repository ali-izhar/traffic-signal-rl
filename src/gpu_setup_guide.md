# Setting Up TensorFlow GPU Support on Windows

This guide provides step-by-step instructions to enable TensorFlow GPU support for your RTX 3060 on Windows.

## 1. Clean Existing Setup

First, let's clean up any existing installations that might be conflicting:

```bash
# Deactivate your current virtual environment if you're using one
deactivate

# Create a fresh virtual environment
python -m venv venv_gpu
.\venv_gpu\Scripts\activate
```

## 2. Install NVIDIA CUDA and cuDNN

For TensorFlow 2.15.0, you need:
- **CUDA Toolkit 11.8**
- **cuDNN 8.6 or higher**

### CUDA Toolkit:
1. Download CUDA Toolkit 11.8 from [NVIDIA Archives](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Select Windows, your architecture (x86_64), and the installer type.
3. Follow the installation instructions and install to the default location.

### cuDNN:
1. Download cuDNN 8.6.0 for CUDA 11.x from [NVIDIA Developer portal](https://developer.nvidia.com/cudnn) (requires free account)
2. Extract and copy files into your CUDA installation directory:
   - Copy `bin\*` files to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - Copy `include\*` files to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
   - Copy `lib\x64\*` files to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64`

## 3. Set Environment Variables

Add or verify these Windows environment variables:

1. Right-click on "This PC" > Properties > Advanced system settings > Environment Variables
2. Add/Update the following System Variables:
   - `CUDA_PATH`: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
   - `Path`: Add entries (if not already present):
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp`
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64`

## 4. Install TensorFlow with GPU Support

Install TensorFlow with the specific versions that work with CUDA 11.8:

```bash
# Install specific versions that work with CUDA 11.8
pip install tensorflow==2.15.0

# Install additional useful packages
pip install gpustat
```

## 5. Verify Installation

Run our diagnostics script to check if GPU is detected:

```bash
python src/check_gpu_diagnostics.py
```

You should see output confirming that your RTX 3060 is recognized.

## 6. Troubleshooting

If your GPU is still not detected:

1. **Check NVIDIA Driver**: Open Device Manager and ensure your RTX 3060 appears without warnings.
2. **Update GPU Driver**: Download latest driver from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)
3. **Check CUDA Installation**: Run `nvcc --version` in command prompt.
4. **Try Restarting Your Computer**: After installing drivers and setting environment variables.
5. **Temporary Environment Variables**: Try setting these before running python:
   ```
   set CUDA_VISIBLE_DEVICES=0
   set TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

## Using GPU with Our Traffic Signal Project

After successful setup, you can run the minimal test:

```bash
python src/test_minimal.py
```

You should see a message about GPU being detected in the output.

For the full training experience on your RTX 3060, modify the configuration by:

```
# Edit src/test_all_agents.py

# Find the "Hardware optimization" section and modify:
config["hardware"] = {
    "gpu": "True",
    "gpu_memory_limit": "6000",  # 6GB for RTX 3060
    "mixed_precision": "True",
    "use_amp": "True", 
    "xla_optimization": "True",
    "num_parallel_calls": "4",
}
```

This optimizes the project to work efficiently with your RTX 3060 GPU. 