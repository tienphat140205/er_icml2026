"""
Router utility functions for format allocation and template application.
"""
import torch
from typing import List, Dict, Tuple

# Format definitions - must match core_algos.py FORMAT_ORDER
FORMAT_NAMES = ['DIRECT', 'COT', 'CODE', 'LONG_COT']
FORMAT_TO_IDX = {name: idx for idx, name in enumerate(FORMAT_NAMES)}


def alloc_format(K: int, probs: torch.Tensor) -> torch.Tensor:
    """
    Allocate budget K across formats by sampling from probability distribution.
    
    Args:
        K: Total number of samples per prompt
        probs: (bsz, num_formats) probability distribution
        
    Returns:
        counts: (bsz, num_formats) integer allocation summing to K
    """
    bsz, num_formats = probs.shape
    
    # Sample K times from categorical distribution
    samples = torch.multinomial(probs, K, replacement=True)  # (bsz, K)
    
    # Count occurrences of each format
    counts = torch.zeros_like(probs, dtype=torch.int32)
    for i in range(bsz):
        counts[i] = torch.bincount(samples[i], minlength=num_formats)
    
    return counts


def get_format_template(format_name: str) -> str:
    """
    Get the prompt template suffix for each format.
    This guides the model to generate in a specific style.
    
    Args:
        format_name: One of 'DIRECT', 'COT', 'CODE', 'LONG_COT'
        
    Returns:
        template: String to append/prepend to the prompt
    """
    templates = {
        'DIRECT': "<ANSWER>",
        'COT': "<COT>",
        'CODE': "<CODE>",
        'LONG_COT': "<LONG_COT>",
    }
    return templates.get(format_name, "")


def apply_format_template(raw_prompt: str, format_name: str) -> str:
    """
    Apply format-specific template to a raw prompt.
    
    Args:
        raw_prompt: Original prompt string
        format_name: One of 'DIRECT', 'SHORT_COT', 'CODE', 'LONG_COT'
        
    Returns:
        formatted_prompt: Prompt with format template applied
    """
    template = get_format_template(format_name)
    # Append format tag after "Assistant:" or at the end
    if "Assistant:" in raw_prompt:
        formatted_prompt = raw_prompt.replace("Assistant:", f"Assistant:{template}")
    else:
        formatted_prompt = raw_prompt + f"\nAssistant:{template}"
    return formatted_prompt


def expand_batch_by_format(
    raw_prompts: List[str],
    format_alloc: torch.Tensor,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Expand prompts based on format allocation.
    Each prompt is duplicated according to its allocated format counts.
    
    Args:
        raw_prompts: List of bsz raw prompt strings
        format_alloc: (bsz, num_formats) allocation counts
        
    Returns:
        expanded_prompts: List of (sum of all allocations) formatted prompts
        prompt_indices: (total,) index mapping back to original prompt
        format_indices: (total,) format index for each expanded prompt
    """
    expanded_prompts = []
    prompt_indices = []
    format_indices = []
    
    bsz = len(raw_prompts)
    for i in range(bsz):
        for f_idx, format_name in enumerate(FORMAT_NAMES):
            n_samples = format_alloc[i, f_idx].item()
            for _ in range(n_samples):
                formatted = apply_format_template(raw_prompts[i], format_name)
                expanded_prompts.append(formatted)
                prompt_indices.append(i)
                format_indices.append(f_idx)
    
    return (
        expanded_prompts,
        torch.tensor(prompt_indices),
        torch.tensor(format_indices)
    )


def apply_format_to_chat(chat: List[dict], format_name: str) -> List[dict]:
    """
    Apply format tag to a chat conversation by adding an assistant turn with the tag.
    Follows the pattern from data preprocessing (e.g., gsm8k.py).
    
    Args:
        chat: List of {"role": "...", "content": "..."} dicts
        format_name: One of 'DIRECT', 'COT', 'CODE', 'LONG_COT'
        
    Returns:
        Modified chat with assistant turn containing format tag
    """
    format_tag = get_format_template(format_name)
    
    # Clone the chat to avoid modifying the original
    chat = [msg.copy() for msg in chat]
    
    # Add assistant turn with format tag (matching preprocessing pattern)
    chat.append({
        "role": "assistant",
        "content": f"{format_tag}\n"
    })
    
    return chat


from verl import DataProto

def prepare_router_rollout_batch(
    batch: DataProto,
    format_alloc: torch.Tensor,
    tokenizer,
    max_prompt_length: int,
) -> DataProto:
    """
    Prepare expanded and tokenized batch for router-guided rollout.
    
    This function:
    1. Expands prompts by format allocation (K samples per prompt)
    2. Applies format tags to each chat
    3. Converts to text using tokenizer's chat template
    4. Tokenizes all expanded prompts
    
    Args:
        batch: DataProto containing 'raw_prompt' in non_tensor_batch
        format_alloc: (bsz, num_formats) allocation counts, should sum to K per row
        tokenizer: HuggingFace tokenizer with apply_chat_template method
        max_prompt_length: Maximum sequence length for tokenization
        
    Returns:
        gen_batch: DataProto containing expanded input_ids, attention_mask, position_ids, etc.
    """
    raw_prompts = batch.non_tensor_batch.get('raw_prompt')
    if raw_prompts is None:
        raise ValueError("batch.non_tensor_batch must contain 'raw_prompt' for router rollout")
    
    expanded_prompts_text = []
    prompt_indices = []
    format_indices = []
    tag_lens = []
    
    bsz = len(raw_prompts)
    for i in range(bsz):
        chat = raw_prompts[i]
        for f_idx, format_name in enumerate(FORMAT_NAMES):
            n_samples = format_alloc[i, f_idx].item()
            for _ in range(n_samples):
                # Add assistant turn with format tag (matching gsm8k.py preprocessing)
                formatted_chat = apply_format_to_chat(chat, format_name)
                
                # Convert to text using tokenizer's chat template
                # continue_final_message=True: continues from assistant message without adding im_end
                # add_generation_prompt=False: don't add another assistant header
                prompt_text = tokenizer.apply_chat_template(
                    formatted_chat, 
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=False
                )
                
                # Calculate tag length in tokens
                format_tag = get_format_template(format_name) + "\n"
                tag_len = len(tokenizer(format_tag, add_special_tokens=False)['input_ids'])
                
                expanded_prompts_text.append(prompt_text)
                prompt_indices.append(i)
                format_indices.append(f_idx)
                tag_lens.append(tag_len)
    
    # Batch tokenize all expanded prompts
    # Force left padding so _pre_process_inputs in vllm_rollout can correctly strip padding
    # (It assumes left padding: finds first non-pad and takes everything after)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    tokenized = tokenizer(
        expanded_prompts_text,
        return_tensors='pt',
        padding='max_length',
        max_length=max_prompt_length,
        truncation=True
    )
    
    # Restore padding side
    tokenizer.padding_side = original_padding_side
    
    # Compute position_ids
    from verl.utils.model import compute_position_id_with_mask
    position_ids = compute_position_id_with_mask(tokenized['attention_mask'])
    
    gen_batch = DataProto.from_dict(
        tensors={
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'position_ids': position_ids,
            'prompt_indices': torch.tensor(prompt_indices),
            'format_indices': torch.tensor(format_indices),
            'tag_lens': torch.tensor(tag_lens)
        },
        meta_info={
            'do_sample': True,
            'n': 1
        }
    )

    return gen_batch



