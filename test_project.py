#!/usr/bin/env python3
"""
Quick test to verify TheraBrush project is working correctly
"""

import os
import sys

def test_environment_variables():
    """Test if .env file is loaded correctly"""
    print("\n" + "="*60)
    print("Testing Environment Variables")
    print("="*60)
    
    from dotenv import load_dotenv
    # Load from pokemon/.env
    load_dotenv('.env')
    
    facepp_key = os.getenv("FACEPP_API_KEY")
    facepp_secret = os.getenv("FACEPP_API_SECRET")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if facepp_key:
        print(f"✅ FACEPP_API_KEY: {facepp_key[:10]}...")
    else:
        print("❌ FACEPP_API_KEY: Not found")
        return False
    
    if facepp_secret:
        print(f"✅ FACEPP_API_SECRET: {facepp_secret[:10]}...")
    else:
        print("❌ FACEPP_API_SECRET: Not found")
        return False
    
    if gemini_key:
        print(f"✅ GEMINI_API_KEY: {gemini_key[:10]}...")
    else:
        print("❌ GEMINI_API_KEY: Not found")
        return False
    
    print("\n✅ All environment variables loaded successfully!")
    return True

def test_imports():
    """Test if key modules can be imported"""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)
    
    # Add current directory to path for imports
    sys.path.insert(0, os.getcwd())
    
    try:
        import main
        print("✅ main.py imports successfully")
        print(f"   - API_KEY loaded: {bool(main.API_KEY)}")
        print(f"   - API_SECRET loaded: {bool(main.API_SECRET)}")
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False
    
    try:
        from chatgpt_advisor import ChatGPTAdvisor
        advisor = ChatGPTAdvisor()
        print("✅ ChatGPTAdvisor initialized")
        print(f"   - API available: {advisor.api_available}")
    except Exception as e:
        print(f"❌ ChatGPTAdvisor failed: {e}")
        return False
    
    try:
        import app_quickdraw
        print("✅ app_quickdraw.py imports successfully")
    except Exception as e:
        print(f"❌ app_quickdraw.py import failed: {e}")
        return False
    
    print("\n✅ All modules imported successfully!")
    return True

def test_model_status():
    """Test if models are ready"""
    print("\n" + "="*60)
    print("Testing Model Status")
    print("="*60)
    
    model_path = "models/drawing_model.keras"
    class_names_path = "models/class_names.txt"
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model found: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  Model not found: {model_path}")
        print("   Run: python train_quickdraw_model.py")
    
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            num_classes = len(f.readlines())
        print(f"✅ Class names found: {num_classes} classes")
    else:
        print(f"⚠️  Class names not found: {class_names_path}")
        print("   Run: python train_quickdraw_model.py")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("THERABRUSH PROJECT TEST")
    print("="*60)
    
    # Change to pokemon directory
    os.chdir(os.path.join(os.path.dirname(__file__), 'pokemon'))
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Module Imports", test_imports),
        ("Model Status", test_model_status),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your project is ready to use.")
        print("\nTo start the application:")
        print("  cd pokemon")
        print("  python app_quickdraw.py")
        print("\nThen open: http://127.0.0.1:5002")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
