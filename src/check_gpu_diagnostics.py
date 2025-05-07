import os
import sys
import subprocess
import platform

print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"System: {platform.system()} {platform.release()}")

# Check if GPU is available through NVIDIA system commands
print("\n=== Checking NVIDIA GPU System Status ===")
try:
    print("\nNVIDIA SMI Output:")
    subprocess.run(["nvidia-smi"], check=False, text=True)
except:
    print("nvidia-smi command failed. NVIDIA driver may not be installed properly.")

# Check TensorFlow installation
print("\n=== Checking TensorFlow Installation ===")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if TensorFlow was built with CUDA support
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # List all available physical devices
    print("\nPhysical devices:")
    for device in tf.config.list_physical_devices():
        print(f" - {device}")
        
    # List available GPUs
    print("\nAvailable GPUs:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f" - {gpu}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"   Details: {details}")
            except:
                print("   Could not get device details")
    else:
        print("No GPUs available to TensorFlow")
    
    # Check if TensorFlow can see CUDA
    print("\nCUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
    
    # Try running a simple operation on GPU to see if it works
    print("\nAttempting to run operation on GPU:")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result: {c}")
            print("GPU operation successful!")
    except Exception as e:
        print(f"GPU operation failed: {e}")
        
except ImportError:
    print("TensorFlow is not installed.")
except Exception as e:
    print(f"Error checking TensorFlow: {e}")

# Check CUDA installation
print("\n=== Checking CUDA Installation ===")
cuda_path = os.environ.get("CUDA_PATH")
if cuda_path:
    print(f"CUDA_PATH environment variable: {cuda_path}")
    if os.path.exists(cuda_path):
        print(f"CUDA directory exists at: {cuda_path}")
        # Check for CUDA version
        nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe" if platform.system() == "Windows" else "nvcc")
        if os.path.exists(nvcc_path):
            try:
                nvcc_output = subprocess.check_output([nvcc_path, "--version"], text=True)
                print(f"NVCC version: {nvcc_output.strip()}")
            except:
                print("Could not determine NVCC version")
    else:
        print(f"CUDA directory does not exist at: {cuda_path}")
else:
    print("CUDA_PATH environment variable is not set")

# Windows-specific checks
if platform.system() == "Windows":
    print("\n=== Windows-Specific Checks ===")
    # Check if any services might be using the GPU
    print("\nChecking for Windows services that might be using the GPU:")
    try:
        subprocess.run(["tasklist", "/FI", "IMAGENAME eq nvwmi.exe"], check=False, text=True)
    except:
        print("Could not check Windows services")

print("\n=== Recommendations ===")
print("If your GPU is not being detected by TensorFlow, try:")
print("1. Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
print("2. Install CUDA Toolkit 11.8 for TensorFlow 2.15: https://developer.nvidia.com/cuda-11-8-0-download-archive")
print("3. Install cuDNN 8.6 for CUDA 11.x: https://developer.nvidia.com/cudnn")
print("4. Reinstall TensorFlow with GPU support: pip install tensorflow==2.15.0")
print("5. Set these environment variables:")
print("   - CUDA_VISIBLE_DEVICES=0")
print("   - TF_FORCE_GPU_ALLOW_GROWTH=true") 