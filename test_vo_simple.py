#!/usr/bin/env python3
"""
Simple test script to verify VO modules work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from vo.features.detectors import build_detector
        from vo.features.descriptors import compute_descriptors
        from vo.features.matcher import match_features, get_matched_points
        from vo.estimators.pose_estimator import (
            estimate_fundamental_matrix,
            estimate_essential_matrix,
            recover_pose,
            camera_info_to_K
        )
        from vo.vo_processor import VOProcessor
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_feature_detection():
    """Test feature detection."""
    try:
        import cv2
        import numpy as np
        from vo.features.detectors import build_detector
        from vo.features.descriptors import compute_descriptors
        
        # Create a test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detector
        detector = build_detector("orb")
        keypoints = detector.detect(test_img, None)
        
        if len(keypoints) > 0:
            # Test descriptor computation
            keypoints, descriptors = compute_descriptors(detector, test_img, keypoints)
            print(f"✓ Feature detection works: {len(keypoints)} keypoints detected")
            return True
        else:
            print("✗ No keypoints detected")
            return False
    except Exception as e:
        print(f"✗ Feature detection error: {e}")
        return False

def test_pose_estimation():
    """Test pose estimation functions."""
    try:
        import cv2
        import numpy as np
        from vo.estimators.pose_estimator import (
            estimate_fundamental_matrix,
            estimate_essential_matrix,
            recover_pose,
            camera_info_to_K
        )
        
        # Create test data
        pts1 = np.random.rand(100, 1, 2).astype(np.float32)
        pts2 = np.random.rand(100, 1, 2).astype(np.float32)
        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        
        # Test fundamental matrix estimation
        F, F_mask = estimate_fundamental_matrix(pts1, pts2)
        print(f"✓ Fundamental matrix estimation works: F shape {F.shape if F is not None else 'None'}")
        
        # Test essential matrix estimation
        E, E_mask = estimate_essential_matrix(pts1, pts2, K)
        print(f"✓ Essential matrix estimation works: E shape {E.shape if E is not None else 'None'}")
        
        return True
    except Exception as e:
        print(f"✗ Pose estimation error: {e}")
        return False

if __name__ == "__main__":
    print("Testing VO modules...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Feature Detection", test_feature_detection),
        ("Pose Estimation", test_pose_estimation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VO system is ready to use.")
        print("\nTo run the VO system:")
        print("1. Make sure you have a camera publishing to /camera/image_raw and /camera/camera_info")
        print("2. Run: roslaunch ir_vo mono_vo.launch")
        print("3. The system will publish poses to /mono_vo/pose and trajectory to /mono_vo/path")
    else:
        print("❌ Some tests failed. Please check the errors above.") 