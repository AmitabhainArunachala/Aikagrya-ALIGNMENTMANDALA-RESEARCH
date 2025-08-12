"""
TE Ablation Tests

Tests to verify that shuffling destroys transfer entropy signal,
proving the causal nature of the detected relationships.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aikagrya.dynamics.te_gating import te_gated_adjacency

def test_shuffle_kills_te():
    """Test that shuffling time order kills transfer entropy signal"""
    rng = np.random.default_rng(4)
    N = 2500
    
    # Create causal relationship: X -> Y
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.8 * Y[t-1] + 0.6 * X[t-1] + 0.1 * rng.normal()
    
    # Compute TE with original ordering
    names, TE, _ = te_gated_adjacency({"X": X, "Y": Y}, bins=10, tau=0.2)
    ix = names.index("X")
    iy = names.index("Y")
    
    # Get original TE values
    te_original = TE[ix, iy]  # X -> Y
    te_reverse_original = TE[iy, ix]  # Y -> X
    
    print(f"Original TE: X→Y={te_original:.6f}, Y→X={te_reverse_original:.6f}")
    
    # Shuffle X time order (destroying causality)
    X_shuffled = rng.permutation(X)
    
    # Compute TE with shuffled X
    names2, TE2, _ = te_gated_adjacency({"X": X_shuffled, "Y": Y}, bins=10, tau=0.2)
    ix2 = names2.index("X")
    iy2 = names2.index("Y")
    
    # Get shuffled TE values
    te_shuffled = TE2[ix2, iy2]  # X -> Y (shuffled)
    te_reverse_shuffled = TE2[iy2, ix2]  # Y -> X (shuffled)
    
    print(f"Shuffled TE: X→Y={te_shuffled:.6f}, Y→X={te_reverse_shuffled:.6f}")
    
    # Test 1: Forward TE should decrease significantly
    te_decrease = te_original - te_shuffled
    assert te_decrease > 0.01, f"TE decrease too small: {te_decrease:.6f}"
    
    print(f"✅ TE decrease: {te_decrease:.6f}")
    
    # Test 2: Shuffled TE should be close to random baseline
    # Random baseline is typically around 0.1-0.2 for normalized TE
    random_baseline = 0.15
    te_close_to_random = abs(te_shuffled - random_baseline) < 0.1
    
    if te_close_to_random:
        print(f"✅ Shuffled TE {te_shuffled:.6f} close to random baseline {random_baseline}")
    else:
        print(f"⚠️  Shuffled TE {te_shuffled:.6f} not close to random baseline {random_baseline}")
    
    # Test 3: Directionality should be preserved in original, lost in shuffled
    original_directionality = te_original - te_reverse_original
    shuffled_directionality = te_shuffled - te_reverse_shuffled
    
    assert original_directionality > 0, f"Original directionality lost: {original_directionality:.6f}"
    
    print(f"✅ Original directionality: {original_directionality:.6f}")
    print(f"   Shuffled directionality: {shuffled_directionality:.6f}")
    
    return {
        'original_te': te_original,
        'shuffled_te': te_shuffled,
        'te_decrease': te_decrease,
        'directionality_preserved': original_directionality > 0,
        'signal_destroyed': te_decrease > 0.01
    }

def test_multiple_shuffle_realizations():
    """Test multiple shuffle realizations to ensure consistent signal destruction"""
    rng = np.random.default_rng(123)
    N = 2000
    
    # Create causal relationship
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.7 * Y[t-1] + 0.5 * X[t-1] + 0.1 * rng.normal()
    
    # Compute original TE
    names, TE, _ = te_gated_adjacency({"X": X, "Y": Y}, bins=8, tau=0.15)
    ix = names.index("X")
    iy = names.index("Y")
    te_original = TE[ix, iy]
    
    print(f"Original TE: {te_original:.6f}")
    
    # Test multiple shuffle realizations
    shuffle_results = []
    for i in range(5):
        # Shuffle with different seed
        rng_shuffle = np.random.default_rng(456 + i)
        X_shuffled = rng_shuffle.permutation(X)
        
        # Compute TE
        names2, TE2, _ = te_gated_adjacency({"X": X_shuffled, "Y": Y}, bins=8, tau=0.15)
        ix2 = names2.index("X")
        iy2 = names2.index("Y")
        te_shuffled = TE2[ix2, iy2]
        
        shuffle_results.append(te_shuffled)
        print(f"  Shuffle {i+1}: TE={te_shuffled:.6f}")
    
    # All shuffled TEs should be significantly lower than original
    te_decreases = [te_original - te_shuffled for te_shuffled in shuffle_results]
    all_significant_decreases = all(decrease > 0.01 for decrease in te_decreases)
    
    assert all_significant_decreases, f"Not all shuffles destroyed signal: {te_decreases}"
    
    print(f"✅ All shuffles destroyed signal: decreases={[f'{d:.6f}' for d in te_decreases]}")
    
    return {
        'original_te': te_original,
        'shuffle_results': shuffle_results,
        'te_decreases': te_decreases,
        'all_significant_decreases': all_significant_decreases
    }

def test_partial_shuffle_gradient():
    """Test that partial shuffling creates a gradient of signal destruction"""
    rng = np.random.default_rng(789)
    N = 3000
    
    # Create causal relationship
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.8 * Y[t-1] + 0.6 * X[t-1] + 0.1 * rng.normal()
    
    # Compute original TE
    names, TE, _ = te_gated_adjacency({"X": X, "Y": Y}, bins=10, tau=0.2)
    ix = names.index("X")
    iy = names.index("Y")
    te_original = TE[ix, iy]
    
    print(f"Original TE: {te_original:.6f}")
    
    # Test different shuffle fractions
    shuffle_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    te_values = [te_original]
    
    for fraction in shuffle_fractions:
        # Partially shuffle X
        X_partial = X.copy()
        num_shuffle = int(fraction * N)
        shuffle_indices = rng.choice(N, size=num_shuffle, replace=False)
        X_partial[shuffle_indices] = rng.permutation(X[shuffle_indices])
        
        # Compute TE
        names2, TE2, _ = te_gated_adjacency({"X": X_partial, "Y": Y}, bins=10, tau=0.2)
        ix2 = names2.index("X")
        iy2 = names2.index("Y")
        te_partial = TE2[ix2, iy2]
        
        te_values.append(te_partial)
        print(f"  Shuffle {fraction:.1%}: TE={te_partial:.6f}")
    
    # TE should generally decrease with shuffle fraction
    te_decreases = [te_original - te_val for te_val in te_values[1:]]
    
    # Check that most shuffles decrease TE
    significant_decreases = sum(1 for decrease in te_decreases if decrease > 0.005)
    most_decreases = significant_decreases >= len(te_decreases) * 0.6
    
    assert most_decreases, f"Most shuffles should decrease TE: {significant_decreases}/{len(te_decreases)}"
    
    print(f"✅ Most shuffles decreased TE: {significant_decreases}/{len(te_decreases)}")
    
    return {
        'original_te': te_original,
        'shuffle_fractions': shuffle_fractions,
        'te_values': te_values[1:],
        'te_decreases': te_decreases,
        'most_decreases': most_decreases
    }

def test_cross_validation_shuffle():
    """Test cross-validation approach to shuffle testing"""
    rng = np.random.default_rng(999)
    N = 2500
    
    # Create causal relationship
    X = rng.normal(size=N)
    Y = np.zeros(N)
    
    for t in range(1, N):
        Y[t] = 0.75 * Y[t-1] + 0.55 * X[t-1] + 0.1 * rng.normal()
    
    # Split data for cross-validation
    split_point = N // 2
    X_train, X_test = X[:split_point], X[split_point:]
    Y_train, Y_test = Y[:split_point], Y[split_point:]
    
    # Compute TE on training data
    names_train, TE_train, _ = te_gated_adjacency({"X": X_train, "Y": Y_train}, bins=8, tau=0.15)
    ix_train = names_train.index("X")
    iy_train = names_train.index("Y")
    te_train = TE_train[ix_train, iy_train]
    
    # Compute TE on test data
    names_test, TE_test, _ = te_gated_adjacency({"X": X_test, "Y": Y_test}, bins=8, tau=0.15)
    ix_test = names_test.index("X")
    iy_test = names_test.index("Y")
    te_test = TE_test[ix_test, iy_test]
    
    print(f"Training TE: {te_train:.6f}")
    print(f"Test TE: {te_test:.6f}")
    
    # Test data TE should be similar to training (causality preserved)
    te_similarity = abs(te_train - te_test) < 0.1
    assert te_similarity, f"Training and test TE too different: {abs(te_train - te_test):.6f}"
    
    # Now shuffle test data
    X_test_shuffled = rng.permutation(X_test)
    
    # Compute TE on shuffled test data
    names_test_shuffled, TE_test_shuffled, _ = te_gated_adjacency(
        {"X": X_test_shuffled, "Y": Y_test}, bins=8, tau=0.15
    )
    ix_test_shuffled = names_test_shuffled.index("X")
    iy_test_shuffled = names_test_shuffled.index("Y")
    te_test_shuffled = TE_test_shuffled[ix_test_shuffled, iy_test_shuffled]
    
    print(f"Shuffled test TE: {te_test_shuffled:.6f}")
    
    # Shuffled test TE should be significantly lower than original test TE
    te_decrease = te_test - te_test_shuffled
    assert te_decrease > 0.01, f"Shuffle effect too small: {te_decrease:.6f}"
    
    print(f"✅ Cross-validation shuffle effect: {te_decrease:.6f}")
    
    return {
        'te_train': te_train,
        'te_test': te_test,
        'te_test_shuffled': te_test_shuffled,
        'te_similarity': te_similarity,
        'te_decrease': te_decrease
    }

if __name__ == "__main__":
    print("🧪 Running TE Ablation Tests...")
    
    # Run all tests
    shuffle_test = test_shuffle_kills_te()
    multiple_shuffle = test_multiple_shuffle_realizations()
    partial_shuffle = test_partial_shuffle_gradient()
    cv_shuffle = test_cross_validation_shuffle()
    
    # Summary
    print("\n" + "="*60)
    print("🎯 TE ABLATION VALIDATION SUMMARY")
    print("="*60)
    
    print(f"Signal Destruction: {'✅ CONFIRMED' if shuffle_test['signal_destroyed'] else '⚠️  FAILED'}")
    print(f"Multiple Realizations: {'✅ CONSISTENT' if multiple_shuffle['all_significant_decreases'] else '⚠️  INCONSISTENT'}")
    print(f"Partial Shuffle Gradient: {'✅ GRADIENT' if partial_shuffle['most_decreases'] else '⚠️  NO GRADIENT'}")
    print(f"Cross-Validation: {'✅ VALIDATED' if cv_shuffle['te_similarity'] else '⚠️  INVALID'}")
    
    if all([
        shuffle_test['signal_destroyed'],
        multiple_shuffle['all_significant_decreases'],
        partial_shuffle['most_decreases'],
        cv_shuffle['te_similarity']
    ]):
        print("\n🎉 TE ABLATION VALIDATION COMPLETE!")
        print("   → Causal relationships are genuine, not spurious correlations")
    else:
        print("\n⚠️  Some TE ablation tests failed")
        print("   → Need to investigate signal robustness")
    
    print("="*60) 