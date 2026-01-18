"""
Test file for compute_router_grpo_outcome_advantage function.
This test simulates the data flow and verifies the rank-based reward scheme.
"""
import torch
import numpy as np
from collections import defaultdict

# Import the function to test
import sys
sys.path.insert(0, '/Users/tienphat/Document/arm/verl')
from verl.trainer.ppo.core_algos import compute_router_grpo_outcome_advantage


def test_router_grpo_advantage():
    """
    Test case:
    - 2 prompts, each with 4 responses (batch_size = 8)
    - Prompt 1: 3 correct responses (lengths: 10, 15, 20), 1 incorrect (length: 12)
    - Prompt 2: 2 correct responses (lengths: 8, 25), 2 incorrect (lengths: 18, 22)
    """
    print("=" * 60)
    print("Testing compute_router_grpo_outcome_advantage")
    print("=" * 60)
    
    batch_size = 8
    max_response_length = 30
    
    # Create mock data
    # token_level_rewards: reward at last valid position (1=correct, 0=incorrect)
    token_level_rewards = torch.zeros(batch_size, max_response_length)
    
    # eos_mask: 1 for valid tokens, 0 for padding
    eos_mask = torch.zeros(batch_size, max_response_length)
    
    # index: prompt IDs (using strings like uuid)
    index = np.array([
        "prompt_1", "prompt_1", "prompt_1", "prompt_1",  # 4 responses for prompt 1
        "prompt_2", "prompt_2", "prompt_2", "prompt_2",  # 4 responses for prompt 2
    ], dtype=object)
    
    # Define response lengths and correctness
    responses = [
        # Prompt 1
        {"length": 10, "correct": True},   # idx 0: shortest correct → rank 0
        {"length": 15, "correct": True},   # idx 1: medium correct → rank 1
        {"length": 20, "correct": True},   # idx 2: longest correct → rank 2
        {"length": 12, "correct": False},  # idx 3: incorrect
        # Prompt 2
        {"length": 8,  "correct": True},   # idx 4: shortest correct → rank 0
        {"length": 25, "correct": True},   # idx 5: longest correct → rank 1
        {"length": 18, "correct": False},  # idx 6: incorrect
        {"length": 22, "correct": False},  # idx 7: incorrect
    ]
    
    # Fill in the mock data
    for i, resp in enumerate(responses):
        length = resp["length"]
        correct = resp["correct"]
        
        # Set eos_mask: 1 for first 'length' positions
        eos_mask[i, :length] = 1.0
        
        # Set reward at last valid position
        if correct:
            token_level_rewards[i, length - 1] = 1.0
    
    print("\n📊 Input Data:")
    print("-" * 40)
    for i, resp in enumerate(responses):
        score = token_level_rewards[i].sum().item()
        length = eos_mask[i].sum().item()
        print(f"  Sample {i}: prompt={index[i]}, length={int(length):2d}, "
              f"correct={resp['correct']}, score={score:.0f}")
    
    # Run the function
    print("\n🔧 Running compute_router_grpo_outcome_advantage...")
    advantages, returns = compute_router_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        eos_mask=eos_mask,
        index=index,
        rho=0.2,
        epsilon=1e-6
    )
    
    # Calculate expected rewards (before normalization)
    print("\n📈 Expected Rank-Based Rewards (before normalization):")
    print("-" * 40)
    rho = 0.2
    expected_rewards = []
    
    # Prompt 1: ranks by length [10, 15, 20] → indices [0, 1, 2]
    # idx 0: rank 0 → 0.2 + 0.8 * (1/2^0) = 1.0
    # idx 1: rank 1 → 0.2 + 0.8 * (1/2^1) = 0.6
    # idx 2: rank 2 → 0.2 + 0.8 * (1/2^2) = 0.4
    # idx 3: incorrect → 0
    expected_rewards.extend([1.0, 0.6, 0.4, 0.0])
    
    # Prompt 2: ranks by length [8, 25] → indices [4, 5]
    # idx 4: rank 0 → 0.2 + 0.8 * (1/2^0) = 1.0
    # idx 5: rank 1 → 0.2 + 0.8 * (1/2^1) = 0.6
    # idx 6: incorrect → 0
    # idx 7: incorrect → 0
    expected_rewards.extend([1.0, 0.6, 0.0, 0.0])
    
    for i, exp in enumerate(expected_rewards):
        print(f"  Sample {i}: expected_reward={exp:.2f}")
    
    # Extract sequence-level advantages (first valid token)
    print("\n✅ Actual Advantages (after normalization):")
    print("-" * 40)
    for i in range(batch_size):
        length = int(eos_mask[i].sum().item())
        # All tokens in a response have same advantage value
        adv_value = advantages[i, 0].item() if length > 0 else 0
        print(f"  Sample {i}: advantage={adv_value:+.4f}")
    
    # Verify key properties
    print("\n🧪 Verification:")
    print("-" * 40)
    
    # Property 1: Correct responses should have higher advantages than incorrect
    # (within same prompt group, after normalization)
    prompt1_correct_advs = [advantages[i, 0].item() for i in [0, 1, 2]]
    prompt1_incorrect_adv = advantages[3, 0].item()
    prompt1_check = all(a > prompt1_incorrect_adv for a in prompt1_correct_advs)
    print(f"  ✓ Prompt 1: All correct > incorrect: {prompt1_check}")
    
    prompt2_correct_advs = [advantages[i, 0].item() for i in [4, 5]]
    prompt2_incorrect_advs = [advantages[i, 0].item() for i in [6, 7]]
    prompt2_check = all(c > i for c in prompt2_correct_advs for i in prompt2_incorrect_advs)
    print(f"  ✓ Prompt 2: All correct > incorrect: {prompt2_check}")
    
    # Property 2: Shorter correct responses should have higher advantages
    prompt1_length_check = advantages[0, 0] > advantages[1, 0] > advantages[2, 0]
    print(f"  ✓ Prompt 1: Shorter correct > longer correct: {prompt1_length_check.item()}")
    
    prompt2_length_check = advantages[4, 0] > advantages[5, 0]
    print(f"  ✓ Prompt 2: Shorter correct > longer correct: {prompt2_length_check.item()}")
    
    # Property 3: Output shape matches input
    shape_check = advantages.shape == token_level_rewards.shape
    print(f"  ✓ Output shape correct: {shape_check}")
    
    # Property 4: Advantages are masked (0 for padding)
    masked_correctly = True
    for i in range(batch_size):
        length = int(eos_mask[i].sum().item())
        if length < max_response_length:
            if advantages[i, length:].sum() != 0:
                masked_correctly = False
    print(f"  ✓ Padding masked to 0: {masked_correctly}")
    
    print("\n" + "=" * 60)
    all_passed = prompt1_check and prompt2_check and prompt1_length_check and prompt2_length_check and shape_check and masked_correctly
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    test_router_grpo_advantage()
