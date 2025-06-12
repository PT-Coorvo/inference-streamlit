# =============================================================================
# STARTUP SCRIPT - run_app.py
# Easy launcher untuk CV Recommendation Streamlit App
# =============================================================================

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path


def check_dependencies():
    """Check apakah semua dependencies sudah terinstall"""
    required_packages = [
        'streamlit>=1.28.0',
        'pandas>=1.5.0', 
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'plotly>=5.15.0'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_name = package.split('>=')[0]
            pkg_resources.require(package)
            print(f"✅ {pkg_name} - OK")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
            print(f"❌ {pkg_name} - Missing")
        except pkg_resources.VersionConflict:
            missing_packages.append(package)
            print(f"⚠️ {pkg_name} - Version conflict")
    
    return missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages"""
    if not missing_packages:
        return True
    
    print(f"\n🔧 Installing missing packages...")
    
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def check_files():
    """Check apakah file-file yang diperlukan ada"""
    required_files = [
        'streamlit_app.py',
        'data_kandidat.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            missing_files.append(file)
            print(f"❌ {file} - Missing")
    
    # Optional files
    optional_files = [
        'cv_system.py',
        'cv_recommendation_model.pkl'
    ]
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found (optional)")
        else:
            print(f"⚠️ {file} - Missing (optional)")
    
    return missing_files

def create_cv_system_stub():
    """Create cv_system.py stub jika tidak ada"""
    if not os.path.exists('cv_system.py'):
        print("📝 Creating cv_system.py stub...")
        
        stub_content = '''
# cv_system.py - STUB FILE
# Copy all functions and classes from your notebook here

print("⚠️ cv_system.py is a stub file!")
print("Please copy all functions and classes from your notebook to this file.")
print("Required components:")
print("- CompleteCVRecommendationSystem class")
print("- load_and_process_data function")
print("- train_content_based_models function")
print("- train_collaborative_filtering_models function") 
print("- create_job_profiles function")
print("- get_hybrid_recommendations function")
print("- save_model function")
print("- load_model function")
print("- main function")

class CompleteCVRecommendationSystem:
    def __init__(self):
        pass

def get_hybrid_recommendations(cv_system, job_id, top_k=5, debug=False):
    return []

def load_model(filename):
    return None

def main():
    return None, {}
'''
        
        with open('cv_system.py', 'w') as f:
            f.write(stub_content)
        
        print("✅ cv_system.py stub created")
        print("⚠️ Please copy your functions from notebook to cv_system.py")

def main():
    """Main startup function"""
    print("="*60)
    print("🚀 CV RECOMMENDATION SYSTEM - STARTUP")
    print("="*60)
    
    print("\n1. 📦 Checking Dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n❌ Found {len(missing_packages)} missing packages")
        install_choice = input("Install missing packages? (y/n): ").lower()
        
        if install_choice == 'y':
            if install_missing_packages(missing_packages):
                print("✅ All packages installed successfully!")
            else:
                print("❌ Package installation failed. Please install manually:")
                for package in missing_packages:
                    print(f"   pip install {package}")
                return
        else:
            print("❌ Cannot run without required packages")
            return
    else:
        print("✅ All dependencies satisfied!")
    
    print("\n2. 📁 Checking Required Files...")
    missing_files = check_files()
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        
        if 'streamlit_app.py' in missing_files:
            print("❌ streamlit_app.py is required!")
            print("Please copy the Streamlit app code to streamlit_app.py")
            return
        
        if 'data_kandidatV2.csv' in missing_files:
            print("⚠️ data_kandidatV2.csv not found")
            print("The app will still run but you'll need to train a new model")
    
    # Create cv_system.py if missing
    if not os.path.exists('cv_system.py'):
        create_cv_system_stub()
        print("\n⚠️ IMPORTANT: Copy your notebook functions to cv_system.py before running!")
        
        proceed = input("Continue anyway? (y/n): ").lower()
        if proceed != 'y':
            return
    
    print("\n3. 🚀 Starting Streamlit App...")
    print("="*60)
    print("🌐 App will open in your browser automatically")
    print("📍 URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("="*60)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed. Please run the script again.")
    except Exception as e:
        print(f"❌ Error running app: {str(e)}")

if __name__ == "__main__":
    main()

# =============================================================================
# BATCH FILE FOR WINDOWS - run_app.bat
# =============================================================================

batch_content = '''@echo off
echo ================================================
echo  CV RECOMMENDATION SYSTEM - WINDOWS LAUNCHER  
echo ================================================

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo Starting CV Recommendation System...
python run_app.py

echo.
echo App finished. Press any key to exit...
pause >nul
'''

# Save batch file for Windows users
with open('run_app.bat', 'w') as f:
    f.write(batch_content)

print("✅ Windows batch file created: run_app.bat")

# =============================================================================
# SHELL SCRIPT FOR LINUX/MAC - run_app.sh  
# =============================================================================

shell_content = '''#!/bin/bash

echo "================================================"
echo " CV RECOMMENDATION SYSTEM - LINUX/MAC LAUNCHER"
echo "================================================"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found! Please install Python 3.8+"
    exit 1
fi

echo "Python version:"
python3 --version

echo "Starting CV Recommendation System..."
python3 run_app.py

echo "App finished."
'''

# Save shell script for Linux/Mac users
with open('run_app.sh', 'w') as f:
    f.write(shell_content)

# Make shell script executable
os.chmod('run_app.sh', 0o755)

print("✅ Linux/Mac shell script created: run_app.sh")