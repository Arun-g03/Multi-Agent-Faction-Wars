import sys
import subprocess
import importlib.util
import traceback

# Path to the requirements file
REQUIREMENTS_FILE = "requirements.txt"

def check_and_install_dependencies():
    """Check for missing dependencies and only install what's missing."""
    try:
        print("[INFO] Checking for missing dependencies...")

        # Read required packages from requirements.txt
        with open(REQUIREMENTS_FILE, "r") as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        missing_packages = []
        for pkg in required_packages:
            pkg_name = pkg.split("==")[0]  # Extract package name without version
            if not importlib.util.find_spec(pkg_name):
                missing_packages.append(pkg)  # Add full package name with version

        if not missing_packages:
            print("[INFO] All dependencies are installed.")
            return True  # No missing dependencies
        
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
