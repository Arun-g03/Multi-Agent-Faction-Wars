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
    """Check for missing dependencies and only install what's missing."""
    try:
        print("[INFO] Checking for missing dependencies...")

        with open(REQUIREMENTS_FILE, "r") as f:
            all_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        pip_lines = all_lines  # We use full lines for pip install
        module_names = [extract_package_name(line) for line in all_lines if not line.startswith("--")]

        missing_packages = []
        for i, pkg_name in enumerate(module_names):
            if pkg_name and not importlib.util.find_spec(pkg_name):
                missing_packages.append(pip_lines[i])  # Use original pip line for install

        if not missing_packages:
            print("[INFO] All dependencies are installed.")
            return True

        print("\033[93mYou may see warnings about missing dependencies that are already installed.\033[0m")
        print("\033[93mInstalling dependencies/libraries is handled by Python Pip. Attempting to install existing dependencies will be skipped.\n\033[0m")
        print("\033[91mYou CANNOT run the program without accepting the install of the flagged dependencies.\n\033[0m")
        print("\033[93m[WARNING] The following dependencies are missing:\033[0m")
        for pkg in missing_packages:
            print(f" - {pkg}")

        choice = input("\nWould you like to install them now? (y/n): ").strip().lower()
        if choice == "y":
            try:
                print("\n[INFO] Installing missing dependencies...\n")
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
                print("\n[INFO] Dependencies installed successfully!")
                return True
            except Exception as e:
                print(f"\n[ERROR] Failed to install dependencies: {e}")
                traceback.print_exc()
                return False
        else:
            print("\033[93m\n[WARNING] Cannot start system without dependencies.\033[0m")
            confirm_exit = input("Are you sure you want to exit? (y/n): ").strip().lower()
            if confirm_exit == "y":
                print("\033[91m\n[ERROR] Missing dependencies detected. Cannot proceed.\033[0m")
                return False
            else:
                return check_and_install_dependencies()

    except FileNotFoundError:
        print(f"\033[91m\n[ERROR] {REQUIREMENTS_FILE} not found. Please create it and list the dependencies.\033[0m")
        return False

if __name__ == "__main__":
    if not check_and_install_dependencies():
        sys.exit("[ERROR] Exiting due to missing dependencies.")
