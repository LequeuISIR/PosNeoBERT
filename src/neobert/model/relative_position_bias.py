import torch
import torch.nn as nn
import math

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance):
        raise NotImplementedError
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # We need (2 * max_distance - 1) slots to cover negative and positive offsets
        self.bias_table = nn.Parameter(
            torch.zeros(num_heads, 2 * max_distance)
        )

    def forward(self, seq_len):
        # 1. Create a matrix of relative distances
        # range(seq_len) looks like, [0, 1, 2]
        pos1 = torch.arange(seq_len, dtype=torch.long, device=self.bias_table.device).view(-1, 1)
        pos2 = torch.arange(seq_len, dtype=torch.long, device=self.bias_table.device).view(1, -1)
        # diffs[i, j] = i - j
        print(pos1)
        print(pos2)
        relative_indices = pos2 - pos1
        
        print("relative_indices")
        print(relative_indices)
        # 2. Shift indices to be non-negative (from 0 to 2*L - 2)
        relative_indices = relative_indices + (self.max_distance)
        print(relative_indices)
        # 3. Index into the bias table 
        # Output shape: (num_heads, seq_len, seq_len)
        return self.bias_table[:, relative_indices]
    


class RelativePositionBucketedBias(nn.Module):
    def __init__(self, num_heads, max_seq_len, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Learnable parameters
        self.relative_attention_bias = nn.Parameter(
            torch.zeros(num_buckets, num_heads)
        )

        # PRE-COMPUTE bucket indices once
        grid_q = torch.arange(max_seq_len, dtype=torch.long).view(-1, 1)
        grid_k = torch.arange(max_seq_len, dtype=torch.long).view(1, -1)
        relative_position = grid_k - grid_q
        
        indices = self._relative_position_bucket(
            relative_position, num_buckets=self.num_buckets, max_distance=self.max_distance
        )

        # Register as a buffer so it moves with the model to GPU
        self.register_buffer("bucket_indices", indices, persistent=True)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        Maps relative distances to bucket indices.
        """
        relative_buckets = 0
        # For bidirectional: use half buckets for negative, half for positive
        # For causal: you'd only care about one direction. 
        # Here we assume bidirectional/symmetric:
        n = -relative_position
        
        num_buckets //= 2
        relative_buckets += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        # Half of the buckets are for 'exact' near distances
        max_exact = num_buckets // 2
        is_small = n < max_exact
        

        # The other half are for logarithmic 'far' distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / 
            math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, n, val_if_large)
        
        return relative_buckets

    def forward(self, seq_len):
        # Create distance grid: (seq_len_q, seq_len_k)
        if seq_len == self.bucket_indices.shape[0]:
            print(2)
            bucket_indices = self.bucket_indices
        else:
            # Fallback to dynamic computation if length changes (e.g., during eval)
            print(1)
            grid_q = torch.arange(seq_len, dtype=torch.long, device=self.relative_attention_bias.device).view(-1, 1)
            grid_k = torch.arange(seq_len, dtype=torch.long, device=self.relative_attention_bias.device).view(1, -1)
            relative_position = grid_k - grid_q

            # Map to buckets
            bucket_indices = self._relative_position_bucket(
                relative_position, num_buckets=self.num_buckets, max_distance=self.max_distance
            )
        

        print(bucket_indices[10])
        
        # Look up biases: (seq_len_q, seq_len_k, num_heads)
        values = self.relative_attention_bias[bucket_indices, :]
        # Permute to (num_heads, seq_len, seq_len) for attention sum
        return values.permute(2, 0, 1).unsqueeze(0) # Adding batch dim if needed
    
if __name__ == "__main__" :
    # pos_bias = RelativePositionBias(1, 8)
    # fw = pos_bias(10)
    # print(fw)

    pos_bias = RelativePositionBucketedBias(12, 512, 32, 128)
    fw = pos_bias(512)
    print(fw.shape)
    
    fw = pos_bias(128)
    print(fw.shape)
    
