import sys
import os
import re
import os
import sys
import subprocess
import traceback
import importlib.util



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")



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

        try:
            with open(REQUIREMENTS_FILE, "r") as f:
                all_lines = [line.strip() for line in f if line.strip()
                                and not line.startswith("#")]
        except FileNotFoundError:
            print(f"[ERROR] Could not find requirements file: {REQUIREMENTS_FILE}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to read requirements file: {e}")
            return False
        pip_lines = all_lines
        module_names = [extract_package_name(
            line) for line in all_lines if not line.startswith("--")]

        missing_packages = []
        for i, pkg_name in enumerate(module_names):
            if pkg_name and not importlib.util.find_spec(pkg_name):
                missing_packages.append(pip_lines[i])

        special_missing = [
            pkg for pkg in [
                "torch",
                "torchvision",
                "torchaudio"] if importlib.util.find_spec(pkg) is None]

        if not missing_packages and not special_missing:
            print("[INFO] All dependencies are installed.")
            return True

        if special_missing:
            print("\n[INFO] Missing PyTorch packages:")
            for pkg in special_missing:
                print(f"- {pkg}")

        if missing_packages:
            print("\n[INFO] Missing other packages:")
            for pkg in missing_packages:
                print(f"- {pkg}")

        user_input = input("\nDo you want to install these missing packages? (y/n): ")
        if user_input.lower() == 'n':
            print("[INFO] Installation cancelled by user.")
            print("Cannot continue without these packages. Exiting...")
            return False
        if special_missing:
            print("\n[INFO] Installing PyTorch packages...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])

        if missing_packages:
            print("\n[INFO] Installing other packages...")
            missing_packages = [pkg for pkg in missing_packages if extract_package_name(
                pkg) not in special_missing]
            if missing_packages:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", *missing_packages])

        print("[INFO] All dependencies installed successfully!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed during dependency installation: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not check_and_install_dependencies():
        sys.exit("[ERROR] Exiting due to missing dependencies.")