import torch
import numpy as np
import time
from typing import Optional

# Import the original implementation for comparison
from model.neural_points import NeuralPoints

# Import the CUDA wrapper (if available)
try:
    from cuda_neural_points.neural_points_cuda_wrapper import (
        NeuralPointsCUDAWrapper,
        create_neural_points,
    )

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA implementation not available")

from utils.config import Config


def create_test_config():
    """
    Create a test configuration"""
    config = Config()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.dtype = torch.float32
    config.feature_dim = 32
    config.feature_std = 0.1
    config.voxel_size_m = 0.1
    config.buffer_size = 1000000
    config.local_map_radius = 10.0
    config.local_map_travel_dist_ratio = 0.1
    config.query_nn_k = 8
    config.num_nei_cells = 2
    config.search_alpha = 10
    config.pos_encoding_band = 0
    config.use_gaussian_pe = False
    config.color_on = False
    config.silence = True
    return config


def benchmark_neural_points():
    """
    Benchmark the performance of neural points implementations"""
    config = create_test_config()

    print("=== NeuralPoints Performance Benchmark ===")
    print(f"Device: {config.device}")
    print(f"Feature dimension: {config.feature_dim}")
    print(f"Voxel size: {config.voxel_size_m}")

    # Generate test data
    n_points = 10000
    n_query_points = 100
    n_frames = 10
    print("Test parameters:")
    print(f"  Points per frame: {n_points}")
    print(f"  Query points: {n_query_points}")
    print(f"  Number of frames: {n_frames}")

    # Test original implementation
    print("\n--- Original Python Implementation ---")
    try:
        neural_points_orig = NeuralPoints(config)

        # Warm up
        for i in range(3):
            points = torch.randn(n_points, 3, device=config.device, dtype=config.dtype)
            sensor_pos = torch.randn(3, device=config.device, dtype=config.dtype)
            sensor_orient = torch.eye(3, device=config.device, dtype=config.dtype)

            neural_points_orig.update(points, sensor_pos, sensor_orient, i)

        # Benchmark update
        start_time = time.time()
        for i in range(n_frames):
            points = torch.randn(n_points, 3, device=config.device, dtype=config.dtype)
            sensor_pos = torch.randn(3, device=config.device, dtype=config.dtype)
            sensor_orient = torch.eye(3, device=config.device, dtype=config.dtype)

            neural_points_orig.update(points, sensor_pos, sensor_orient, i)

        update_time_orig = time.time() - start_time
        print(f"Update time: {update_time_orig:.4f}s")

        # Benchmark query
        query_points = torch.randn(
            n_query_points, 3, device=config.device, dtype=config.dtype
        )

        start_time = time.time()
        for _ in range(10):
            geo_features, _, weights, nn_counts, certainty = (
                neural_points_orig.query_feature(query_points, training_mode=False)
            )

        query_time_orig = time.time() - start_time
        print(f"Query time: {query_time_orig:.4f}s")
        print(f"Total neural points: {neural_points_orig.count()}")

    except Exception as e:
        print(f"Original implementation failed: {e}")
        update_time_orig = float("inf")
        query_time_orig = float("inf")

    # Test CUDA implementation
    if CUDA_AVAILABLE:
        print("\n--- CUDA Implementation ---")
        try:
            neural_points_cuda = create_neural_points(config)

            # Warm up
            for i in range(3):
                points = torch.randn(
                    n_points, 3, device=config.device, dtype=config.dtype
                )
                sensor_pos = torch.randn(3, device=config.device, dtype=config.dtype)
                sensor_orient = torch.eye(3, device=config.device, dtype=config.dtype)

                neural_points_cuda.update(points, sensor_pos, sensor_orient, i)

            # Benchmark update
            start_time = time.time()
            for i in range(n_frames):
                points = torch.randn(
                    n_points, 3, device=config.device, dtype=config.dtype
                )
                sensor_pos = torch.randn(3, device=config.device, dtype=config.dtype)
                sensor_orient = torch.eye(3, device=config.device, dtype=config.dtype)

                neural_points_cuda.update(points, sensor_pos, sensor_orient, i)

            update_time_cuda = time.time() - start_time
            print(f"Update time: {update_time_cuda:.4f}s")

            # Benchmark query
            query_points = torch.randn(
                n_query_points, 3, device=config.device, dtype=config.dtype
            )

            start_time = time.time()
            for _ in range(10):
                geo_features, _, weights, nn_counts, certainty = (
                    neural_points_cuda.query_feature(query_points, training_mode=False)
                )

            query_time_cuda = time.time() - start_time
            print(f"Query time: {query_time_cuda:.4f}s")
            print(f"Total neural points: {neural_points_cuda.count()}")

        except Exception as e:
            print(f"CUDA implementation failed: {e}")
            update_time_cuda = float("inf")
            query_time_cuda = float("inf")
    else:
        print("CUDA implementation not available")
        update_time_cuda = float("inf")
        query_time_cuda = float("inf")

    # Performance comparison
    print("\n--- Performance Comparison ---")
    if update_time_orig != float("inf") and update_time_cuda != float("inf"):
        speedup_update = update_time_orig / update_time_cuda
        print(f"Update speedup: {speedup_update:.2f}x")

    if query_time_orig != float("inf") and query_time_cuda != float("inf"):
        speedup_query = query_time_orig / query_time_cuda
        print(f"Query speedup: {speedup_query:0.2f}x")


def test_memory_usage():
    """
    Test memory usage of different implementations"""
    config = create_test_config()

    print("\n=== Memory Usage Test ===")
    # Test with larger dataset
    n_points = 5000
    n_frames = 20

    print(f"Testing with {n_points} points per frame for {n_frames} frames")

    # Original implementation
    print("\n--- Original Implementation Memory ---")
    try:
        neural_points_orig = NeuralPoints(config)

        for i in range(n_frames):
            points = torch.randn(n_points, 3, device=config.device, dtype=config.dtype)
            sensor_pos = torch.randn(3, device=config.device, dtype=config.dtype)
            sensor_orient = torch.eye(3, device=config.device, dtype=config.dtype)

            neural_points_orig.update(points, sensor_pos, sensor_orient, i)

            if i % 5 == 0:
                neural_points_orig.record_memory()
                print(f"Frame {i}: {neural_points_orig.count()} points")

    except Exception as e:
        print(f"Original implementation failed: {e}")

    # CUDA implementation
    if CUDA_AVAILABLE:
        print("\n--- CUDA Implementation Memory ---")
        try:
            neural_points_cuda = create_neural_points(config)

            for i in range(n_frames):
                points = torch.randn(
                    n_points, 3, device=config.device, dtype=config.dtype
                )
                sensor_pos = torch.randn(3, device=config.device, dtype=config.dtype)
                sensor_orient = torch.eye(3, device=config.device, dtype=config.dtype)

                neural_points_cuda.update(points, sensor_pos, sensor_orient, i)

                if i % 5 == 0:
                    neural_points_cuda.record_memory()
                    print(f"Frame {i}: {neural_points_cuda.count()} points")

        except Exception as e:
            print(f"CUDA implementation failed: {e}")


def test_accuracy():
    """
    Test accuracy between implementations"""
    config = create_test_config()

    print("\n=== Accuracy Test ===")

    # Use deterministic data
    torch.manual_seed(42)
    n_points = 1000
    n_query_points = 100

    # Generate test data
    points = torch.randn(n_points, 3, device=config.device, dtype=config.dtype)
    sensor_pos = torch.randn(3, device=config.device, dtype=config.dtype)
    sensor_orient = torch.eye(3, device=config.device, dtype=config.dtype)
    query_points = torch.randn(
        n_query_points, 3, device=config.device, dtype=config.dtype
    )

    # Test original implementation
    print("Testing original implementation...")
    try:
        neural_points_orig = NeuralPoints(config)
        neural_points_orig.update(points, sensor_pos, sensor_orient, 0)

        geo_features_orig, _, weights_orig, nn_counts_orig, certainty_orig = (
            neural_points_orig.query_feature(query_points, training_mode=False)
        )

        print(f"Original - Features shape: {geo_features_orig.shape}")
        print(f"Original - Weights shape: {weights_orig.shape}")
        print(f"Original - NN counts: {nn_counts_orig.shape}")

    except Exception as e:
        print(f"Original implementation failed: {e}")
        geo_features_orig = None

    # Test CUDA implementation
    if CUDA_AVAILABLE:
        print("Testing CUDA implementation...")
        try:
            neural_points_cuda = create_neural_points(config)
            neural_points_cuda.update(points, sensor_pos, sensor_orient, 0)

            geo_features_cuda, _, weights_cuda, nn_counts_cuda, certainty_cuda = (
                neural_points_cuda.query_feature(query_points, training_mode=False)
            )

            print(f"CUDA - Features shape: {geo_features_cuda.shape}")
            print(f"CUDA - Weights shape: {weights_cuda.shape}")
            print(f"CUDA - NN counts: {nn_counts_cuda.shape}")

            # Compare results if both succeeded
            if geo_features_orig is not None:
                print("\n--- Accuracy Comparison ---")

                # Compare feature shapes
                if geo_features_orig.shape == geo_features_cuda.shape:
                    print("✓ Feature shapes match")
                else:
                    print("✗ Feature shapes differ")

                # Compare weights shapes
                if weights_orig.shape == weights_cuda.shape:
                    print("✓ Weight shapes match")
                else:
                    print("✗ Weight shapes differ")

                # Compare NN counts
                if nn_counts_orig.shape == nn_counts_cuda.shape:
                    print("✓ NN count shapes match")
                else:
                    print("✗ NN count shapes differ")

        except Exception as e:
            print(f"CUDA implementation failed: {e}")


if __name__ == "__main__":
    print("Neural Points CUDA Implementation Test")
    print("=" * 50)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available")

    # Run tests
    benchmark_neural_points()
    test_memory_usage()
    test_accuracy()

    print("\nTest completed!")
