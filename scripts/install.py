#!/usr/bin/env python
"""
NeuroSync-Core Installation Script
This script automatically detects your CUDA capabilities and installs the 
appropriate version of PyTorch and other dependencies.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {platform.python_version()}")

def check_hardware_acceleration():
    """Check for hardware acceleration availability (CUDA, MPS, or CPU)"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            # Check if we're on Apple Silicon
            output = subprocess.check_output(['uname', '-m'], universal_newlines=True).strip()
            if output in ['arm64']:
                print("✅ Apple Silicon detected - MPS acceleration available")
                return "mps"
            else:
                print("⚠️ Intel Mac detected - CPU only")
                return "cpu"
        except Exception:
            print("⚠️ Could not determine Mac architecture - defaulting to CPU")
            return "cpu"
    else:
        # Check for NVIDIA CUDA on Linux/Windows
        try:
            output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
            for line in output.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip()
                    print(f"✅ CUDA detected: {cuda_version}")
                    return cuda_version
            print("❌ NVIDIA GPU detected but couldn't determine CUDA version")
            return "cpu"
        except Exception:
            print("❌ No NVIDIA GPU detected or drivers not installed")
            return "cpu"

def install_dependencies(hardware_acceleration):
    """Install dependencies based on hardware acceleration availability"""
    print("\n📦 Installing dependencies...")
    
    # Install base requirements
    requirements = [
        "numpy",
        "scipy",
        "flask",
        "pydub",
        "sounddevice",
        "transformers",
        "librosa",
    ]
    
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + requirements)
    print("✅ Installed base dependencies")
    
    # Install PyTorch based on hardware acceleration type
    if hardware_acceleration == "mps":
        print("🚀 Installing PyTorch with MPS (Metal) support for Apple Silicon")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'])
        return
    elif hardware_acceleration == "cpu":
        print("⚠️ Installing CPU-only PyTorch version (slower)")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'torch', 'torchvision', 'torchaudio'])
        return
    
    # Handle CUDA versions for Linux/Windows
    try:
        cuda_major_minor = '.'.join(hardware_acceleration.split('.')[:2])
        cuda_float = float(cuda_major_minor)
    except Exception:
        print(f"⚠️ Could not parse CUDA version: {hardware_acceleration}, installing CPU version")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'torch', 'torchvision', 'torchaudio'])
        return
    
    # Install PyTorch based on CUDA version
    if cuda_float >= 12.8:
        print("🚀 Installing PyTorch with CUDA 12.8 support (nightly build)")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torch', 
                                  '--index-url', 'https://download.pytorch.org/whl/nightly/cu128'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torchvision', 
                                  '--index-url', 'https://download.pytorch.org/whl/nightly/cu128'])
        except Exception as e:
            print(f"⚠️ Failed to install nightly build: {e}")
            print("⚠️ Falling back to stable CUDA 12.1")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                                  '--index-url', 'https://download.pytorch.org/whl/cu121'])
    elif cuda_float >= 12.1:
        print("🚀 Installing PyTorch with CUDA 12.1 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu121'])
    elif cuda_float >= 11.8:
        print("🚀 Installing PyTorch with CUDA 11.8 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu118'])
    elif cuda_float >= 11.7:
        print("🚀 Installing PyTorch with CUDA 11.7 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu117'])
    else:
        print(f"⚠️ CUDA {hardware_acceleration} is older than recommended. Installing CPU version")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'])

def verify_installation():
    """Verify PyTorch is installed correctly with hardware acceleration if available"""
    print("\n🔍 Verifying installation...")
    try:
        verification_script = """
import torch
import platform
print(f"PyTorch version: {torch.__version__}")
print(f"Platform: {platform.system()} {platform.machine()}")

if platform.system() == "Darwin":  # macOS
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    if torch.backends.mps.is_available():
        # Test MPS tensor operation
        try:
            x = torch.rand(5, 3, device='mps')
            y = torch.rand(3, 5, device='mps')
            z = x @ y
            print("✅ MPS (Metal) tensor operation successful!")
        except Exception as e:
            print(f"⚠️ MPS tensor operation failed: {e}")
            print("⚠️ Falling back to CPU")
    else:
        print("⚠️ MPS not available, using CPU only")
else:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        # Test CUDA tensor operation
        try:
            x = torch.rand(5, 3, device='cuda')
            y = torch.rand(3, 5, device='cuda')
            z = x @ y
            print("✅ CUDA tensor operation successful!")
        except Exception as e:
            print(f"⚠️ CUDA tensor operation failed: {e}")
    else:
        print("⚠️ CUDA not available, using CPU only")

# Test basic CPU operation
x = torch.rand(5, 3)
y = torch.rand(3, 5)
z = x @ y
print("✅ CPU tensor operation successful!")
"""
        subprocess.run([sys.executable, '-c', verification_script], check=True)
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False
    
    return True

def main():
    print("=" * 70)
    print("🧠 NeuroSync-Core Installation Script")
    print("=" * 70)
    
    check_python_version()
    hardware_acceleration = check_hardware_acceleration()
    install_dependencies(hardware_acceleration)
    
    success = verify_installation()
    
    if success:
        print("\n✅ Installation completed successfully!")
        print("\n📝 Next steps:")
        print("1. Run 'python -m neurosync.server.app' to start the server")
        print("2. Check out the documentation for more information")
        
        # Display hardware acceleration info
        if hardware_acceleration == "mps":
            print("🚀 Using Apple Metal Performance Shaders for acceleration")
        elif hardware_acceleration == "cpu":
            print("⚠️ Using CPU-only processing (slower but compatible)")
        else:
            print(f"🚀 Using CUDA {hardware_acceleration} for acceleration")
    else:
        print("\n❌ Installation may not be complete. Please check the errors above.")
        print("If you're having issues, try installing manually:")
        if platform.system() == "Darwin":
            print("pip install torch torchvision torchaudio")
        else:
            print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main() 