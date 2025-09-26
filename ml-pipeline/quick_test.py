#!/usr/bin/env python3
"""
Quick test script to verify everything is working
Run this after AI completes your code components
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 
        'pydicom', 'SimpleITK', 'monai', 'matplotlib'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - FAILED")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {failed_imports}")
        print("Run: pip install -r requirements.txt")
    else:
        print("\n🎉 All required packages imported successfully!")
    
    return len(failed_imports) == 0

def test_project_structure():
    """Test if project structure is correct"""
    print("\n🔍 Testing project structure...")
    
    required_dirs = [
        'src/data', 'src/models', 'src/training', 'src/utils',
        'config', 'test_components', 'notebooks', 'logs'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {missing_dirs}")
    else:
        print("\n🎉 Project structure is correct!")
    
    return len(missing_dirs) == 0

def test_config_files():
    """Test if configuration files exist"""
    print("\n🔍 Testing configuration files...")
    
    config_files = ['config/training_config.yaml', 'requirements.txt']
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - MISSING")

def main():
    print("🚀 NSCLC Project Quick Test")
    print("=" * 40)
    
    imports_ok = test_imports()
    structure_ok = test_project_structure()
    test_config_files()
    
    print("\n" + "=" * 40)
    if imports_ok and structure_ok:
        print("🎉 Setup complete! Ready for AI code generation.")
        print("\nNext steps:")
        print("1. Use AI to complete the code templates")
        print("2. Run: python run_local_tests.py")
        print("3. Test with sample data")
        print("4. Deploy to Colab")
    else:
        print("⚠️  Setup incomplete. Fix the issues above first.")

if __name__ == "__main__":
    main()
EOFcat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test script to verify everything is working
Run this after AI completes your code components
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 
        'pydicom', 'SimpleITK', 'monai', 'matplotlib'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - FAILED")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {failed_imports}")
        print("Run: pip install -r requirements.txt")
    else:
        print("\n🎉 All required packages imported successfully!")
    
    return len(failed_imports) == 0

def test_project_structure():
    """Test if project structure is correct"""
    print("\n🔍 Testing project structure...")
    
    required_dirs = [
        'src/data', 'src/models', 'src/training', 'src/utils',
        'config', 'test_components', 'notebooks', 'logs'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {missing_dirs}")
    else:
        print("\n🎉 Project structure is correct!")
    
    return len(missing_dirs) == 0

def test_config_files():
    """Test if configuration files exist"""
    print("\n🔍 Testing configuration files...")
    
    config_files = ['config/training_config.yaml', 'requirements.txt']
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - MISSING")

def main():
    print("🚀 NSCLC Project Quick Test")
    print("=" * 40)
    
    imports_ok = test_imports()
    structure_ok = test_project_structure()
    test_config_files()
    
    print("\n" + "=" * 40)
    if imports_ok and structure_ok:
        print("🎉 Setup complete! Ready for AI code generation.")
        print("\nNext steps:")
        print("1. Use AI to complete the code templates")
        print("2. Run: python run_local_tests.py")
        print("3. Test with sample data")
        print("4. Deploy to Colab")
    else:
        print("⚠️  Setup incomplete. Fix the issues above first.")

if __name__ == "__main__":
    main()
