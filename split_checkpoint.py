import torch
import os

def split_checkpoint(checkpoint_path, output_dir, chunk_size_mb=23):
    """Splits a PyTorch checkpoint into smaller chunks."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    state_dict = checkpoint.get('model_state_dict', checkpoint)

    tensors = sorted(state_dict.items())

    current_chunk_tensors = {}
    current_chunk_size = 0
    chunk_num = 1

    for name, tensor in tensors:
        tensor_size = tensor.nelement() * tensor.element_size()

        if current_chunk_tensors and current_chunk_size + tensor_size > chunk_size_mb * 1024 * 1024:
            chunk_filename = os.path.join(output_dir, f'checkpoint_part_{chunk_num:02d}.pth')
            torch.save(current_chunk_tensors, chunk_filename)
            print(f'Saved {chunk_filename}')
            chunk_num += 1
            current_chunk_tensors = {}
            current_chunk_size = 0

        current_chunk_tensors[name] = tensor
        current_chunk_size += tensor_size

    if current_chunk_tensors:
        chunk_filename = os.path.join(output_dir, f'checkpoint_part_{chunk_num:02d}.pth')
        torch.save(current_chunk_tensors, chunk_filename)
        print(f'Saved {chunk_filename}')

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_file = os.path.join(base_dir, 'pretrained_unet_checkpoint.pth')
    output_directory = os.path.join(base_dir, 'split_checkpoint')

    split_checkpoint(checkpoint_file, output_directory)
    print("Checkpoint splitting complete.")