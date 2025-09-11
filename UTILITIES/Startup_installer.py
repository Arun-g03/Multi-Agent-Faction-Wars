import sys
import os
import re
import subprocess
import traceback
import importlib.util
import pkg_resources
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")

# Package name mappings (pip name -> import name)
PACKAGE_MAPPINGS = {
    'opencv-python': 'cv2',
    'scikit-learn': 'sklearn',
    'PIL': 'PIL',  # Pillow
}

# Alternative import names for common packages
ALTERNATIVE_IMPORTS = {
    'cv2': ['cv2', 'opencv'],
    'sklearn': ['sklearn', 'sklearn.utils', 'sklearn.ensemble'],
    'PIL': ['PIL', 'PIL.Image', 'PIL.ImageDraw'],
    'torch': ['torch', 'torch.nn', 'torch.optim'],
    'torchvision': ['torchvision', 'torchvision.transforms'],
    'torchaudio': ['torchaudio', 'torchaudio.transforms'],
}

def extract_package_info(requirement_line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts package name, version, and operator from a requirement line.
    Returns: (package_name, operator, version)
    """
    if requirement_line.startswith("--") or requirement_line.startswith("#"):
        return None, None, None
    
    # Handle complex requirements like "torch==2.2.0+cu121"
    match = re.match(r"^([a-zA-Z0-9_-]+)\s*([<=>!~]+)\s*(.+)$", requirement_line.strip())
    if match:
        return match.group(1), match.group(2), match.group(3)
    
    # Handle simple package names
    package_name = requirement_line.strip()
    if package_name:
        return package_name, None, None
    
    return None, None, None

def get_installed_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package using multiple methods."""
    try:
        # Method 1: Try pkg_resources
        try:
            return pkg_resources.get_distribution(package_name).version
        except:
            pass
        
        # Method 2: Try importlib.metadata (Python 3.8+)
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except:
            pass
        
        # Method 3: Try pip list
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], 
                                 capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if line.startswith(f"{package_name}=="):
                    return line.split('==')[1]
        except:
            pass
            
    except Exception:
        pass
    
    return None

def check_package_importable(package_name: str, import_name: str = None) -> bool:
    """
    Check if a package can be imported using multiple methods.
    
    Args:
        package_name: The pip package name
        import_name: The import name (if different from package name)
    
    Returns:
        bool: True if package can be imported
    """
    if import_name is None:
        import_name = package_name
    
    # Try direct import first
    try:
        if importlib.util.find_spec(import_name):
            return True
    except:
        pass
    
    # Try alternative import names
    if import_name in ALTERNATIVE_IMPORTS:
        for alt_name in ALTERNATIVE_IMPORTS[import_name]:
            try:
                if importlib.util.find_spec(alt_name):
                    return True
            except:
                continue
    
    # Try importing specific submodules
    try:
        # For packages like torch, try importing a submodule
        if import_name == 'torch':
            import torch
            return True
        elif import_name == 'torchvision':
            import torchvision
            return True
        elif import_name == 'torchaudio':
            import torchaudio
            return True
        elif import_name == 'cv2':
            import cv2
            return True
        elif import_name == 'sklearn':
            import sklearn
            return True
        elif import_name == 'PIL':
            import PIL
            return True
    except ImportError:
        pass
    
    return False

def check_version_compatibility(required_version: str, installed_version: str, operator: str) -> bool:
    """
    Check if installed version meets the requirement.
    
    Args:
        required_version: The required version string
        installed_version: The installed version string
        operator: The comparison operator (==, >=, <=, etc.)
    
    Returns:
        bool: True if version is compatible
    """
    if not required_version or not installed_version:
        return True  # Can't check, assume compatible
    
    try:
        # Clean version strings (remove +cu118, +cu121, etc.)
        clean_req = re.sub(r'\+[a-zA-Z0-9]+', '', required_version)
        clean_inst = re.sub(r'\+[a-zA-Z0-9]+', '', installed_version)
        
        # Parse versions
        from packaging import version
        req_ver = version.parse(clean_req)
        inst_ver = version.parse(clean_inst)
        
        if operator == '==':
            return req_ver == inst_ver
        elif operator == '>=':
            return inst_ver >= req_ver
        elif operator == '<=':
            return inst_ver <= req_ver
        elif operator == '>':
            return inst_ver > req_ver
        elif operator == '<':
            return inst_ver < req_ver
        elif operator == '!=':
            return inst_ver != req_ver
        elif operator == '~=':
            # Compatible release (same major version)
            return inst_ver >= req_ver and inst_ver < version.parse(f"{req_ver.major + 1}.0.0")
        else:
            return True  # Unknown operator, assume compatible
            
    except Exception as e:
        print(f"[WARNING] Could not check version compatibility for {required_version}: {e}")
        return True  # Can't check, assume compatible

def check_and_install_dependencies() -> bool:
    """
    Enhanced dependency checker that handles version mismatches and package variations.
    """
    try:
        print("[INFO] Checking dependencies with enhanced detection...")
        print("[INFO] This may take a moment...")

        # Read requirements file
        try:
            with open(REQUIREMENTS_FILE, "r") as f:
                all_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except FileNotFoundError:
            print(f"[ERROR] Could not find requirements file: {REQUIREMENTS_FILE}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to read requirements file: {e}")
            return False

        missing_packages = []
        version_mismatches = []
        working_packages = []
        
        print(f"[INFO] Checking {len(all_lines)} package requirements...")

        for i, line in enumerate(all_lines):
            package_name, operator, version = extract_package_info(line)
            
            if not package_name:
                continue
                
            print(f"[INFO] Checking {package_name}...", end=" ")
            
            # Get the import name
            import_name = PACKAGE_MAPPINGS.get(package_name, package_name)
            
            # Check if package is importable
            if check_package_importable(package_name, import_name):
                # Package is importable, check version
                installed_version = get_installed_version(package_name)
                
                if installed_version:
                    print(f"✓ Installed (v{installed_version})")
                    
                    # Check version compatibility
                    if version and not check_version_compatibility(version, installed_version, operator):
                        version_mismatches.append({
                            'package': package_name,
                            'required': version,
                            'installed': installed_version,
                            'operator': operator,
                            'full_line': line
                        })
                        print(f"  ⚠ Version mismatch: {operator}{version} vs {installed_version}")
                    else:
                        working_packages.append(package_name)
                else:
                    print("✓ Installed (version unknown)")
                    working_packages.append(package_name)
            else:
                print("✗ Not found")
                missing_packages.append(line)

        # Summary
        print(f"\n[INFO] Dependency Check Summary:")
        print(f"✓ Working packages: {len(working_packages)}")
        print(f"⚠ Version mismatches: {len(version_mismatches)}")
        print(f"✗ Missing packages: {len(missing_packages)}")
        
        if working_packages:
            print(f"\n[INFO] Working packages: {', '.join(working_packages[:10])}{'...' if len(working_packages) > 10 else ''}")
        
        # Handle version mismatches
        if version_mismatches:
            print(f"\n[WARNING] Version mismatches detected:")
            for mismatch in version_mismatches:
                print(f"  {mismatch['package']}: {mismatch['operator']}{mismatch['required']} required, {mismatch['installed']} installed")
            
            user_input = input("\nSome packages have version mismatches. Continue anyway? (y/n): ")
            if user_input.lower() != 'y':
                print("[INFO] Installation cancelled by user.")
                return False

        # Handle missing packages
        if missing_packages:
            print(f"\n[INFO] Missing packages:")
            for pkg in missing_packages:
                print(f"  - {pkg}")
            
            user_input = input("\nDo you want to install these missing packages? (y/n): ")
            if user_input.lower() == 'n':
                print("[INFO] Installation cancelled by user.")
                print("Cannot continue without these packages. Exiting...")
                return False
            
            # Install missing packages
            print("\n[INFO] Installing missing packages...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
                print("[INFO] All missing packages installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to install packages: {e}")
                return False
        else:
            print("[INFO] All required packages are available!")

        return True

    except Exception as e:
        print(f"\n[ERROR] Failed during dependency installation: {e}")
        traceback.print_exc()
        return False

def test_imports():
    """Test importing key packages to verify they work."""
    print("\n[INFO] Testing package imports...")
    
    test_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('torchaudio', 'torchaudio'),
        ('numpy', 'numpy'),
        ('pygame', 'pygame'),
        ('cv2', 'cv2'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
    ]
    
    failed_imports = []
    
    for package_name, import_name in test_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} imports successfully")
        except ImportError as e:
            print(f"✗ {package_name} failed to import: {e}")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"\n[WARNING] {len(failed_imports)} packages failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n[SUCCESS] All test packages imported successfully!")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Dependency Checker for MultiAgent Faction Wars")
    print("=" * 60)
    
    if check_and_install_dependencies():
        print("\n[INFO] Running import tests...")
        if test_imports():
            print("\n[SUCCESS] All dependencies are working correctly!")
        else:
            print("\n[WARNING] Some packages have import issues but may still work.")
    else:
        print("\n[ERROR] Dependency check failed. Exiting...")
        sys.exit(1)