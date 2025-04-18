import sys
import subprocess
import importlib.util
import traceback
import re

REQUIREMENTS_FILE = "requirements.txt"

def extract_package_name(requirement_line):
    """
    Extracts the base package name (e.g., 'torch' from 'torch==2.2.0+cu121')
    """
    if requirement_line.startswith("--"):
        return None
    return re.split(r"[<=>]", requirement_line)[0].strip()

def check_and_install_dependencies():
    try:
        print("[INFO] Checking for missing dependencies...")

        with open(REQUIREMENTS_FILE, "r") as f:
            all_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        pip_lines = all_lines
        module_names = [extract_package_name(line) for line in all_lines if not line.startswith("--")]

        missing_packages = []
        for i, pkg_name in enumerate(module_names):
            if pkg_name and not importlib.util.find_spec(pkg_name):
                missing_packages.append(pip_lines[i])

        """Check if the the pytorch machine learning components are missing: torch, torchvision, torchaudio
            and install with the correct command:
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        """
        special_missing = [pkg for pkg in ["torch", "torchvision", "torchaudio"]
                           if importlib.util.find_spec(pkg) is None]

        if special_missing:
            print("\n[INFO] Detected missing torch-related packages. Installing from PyTorch custom index...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            # Remove torch-related entries to avoid duplicate install
            missing_packages = [pkg for pkg in missing_packages if extract_package_name(pkg) not in special_missing]

        if not missing_packages:
            print("[INFO] All other dependencies are installed.")
            return True

        print("\n[INFO] Installing other missing dependencies...\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        print("[INFO] Dependencies installed successfully!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed during dependency installation: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not check_and_install_dependencies():
        sys.exit("[ERROR] Exiting due to missing dependencies.")
