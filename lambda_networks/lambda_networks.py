import torch
from torch import nn, einsum
from einops import rearrange

# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    # torch.stack(pos).shape == (2, n, n)
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    # pos: (2, n, n) -> (n*n, 2)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    '''
        There are n*n positions in the form (i, j)
        (because this is an image if it is a sequence, there are only n positions)
        Let N = n*n and (q, r) be the index for rel_pos
        0 <= q, r < N
        rel_pos[q][r] = pos[q] - pos[r]
        and rel_pos[q][r].shape == (2, )

        After the value shifting later,
        a rel_pos[q][r] == (rel_i, rel_j) will be used to index the parameter
    '''
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer

class LambdaLayer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_k,
        n = None,
        r = None,
        heads = 4,
        dim_out = None,
        dim_u = 1):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            # Positional embeddings are directly encoded in this Conv3d
            # The padding will gaurantee the input height and width are the same
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        else:
            assert exists(n), 'You must specify the window size (n=h=w)'
            rel_lengths = 2 * n - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
            self.rel_pos = calc_rel_pos(n)

    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x) # multiquery: there are |heads| of (k, ) query
        k = self.to_k(x) # each pixel has a (k, u) feature matrix represent "key"
        v = self.to_v(x) # each pixel has a (v, u) feature matrix represent "value"

        # Stated in the last part of 3.1
        q = self.norm_q(q)
        v = self.norm_v(v)

        # Flatten the image (hh, ww) into a single dimension
        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        # Normalize across positions (in this case, it will normalize across pixels)
        k = k.softmax(dim=-1)

        # I will think of (b, u, k, m) (b, u, v, m) -> (b, k, v)
        # as a tensor product of (b, u, m, k) and (b, u, m, v)
        # Note that, (b, u, m) has the same dimension, so only k, v are broadcasted
        # After the tensor product, it will produce (b, u, m, k, v)
        # And because u, m are not specified, it will be summed up to produce (b, k, v)
        # (Can also easily achieved by (b, u, m, k, v) -> (b, u*m, k, v) -> (b, k, v)
        λc = einsum('b u k m, b u v m -> b k v', k, v)
        # Time complexity: O(b * k) (same dimension) * O(hnv) (broadcast multiplication) = O(b k h n v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        if self.local_contexts:
            # u is the channel we want it be transformed to k,
            # so the author put it at the channel axis.
            v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            # (b, u, v, hh, ww) -> (b, k, v, hh, ww)
            # For each pixel, there is a lambda function (i.e. (k, v) matrix)
            λp = self.pos_conv(v)
            # Apply the lambda function at each pixel w.r.t the query
            # Broadcast multiplication: (b, k, n, h) * (b, k, n, v) -> (b, k, n, h, v)
            # Contract k -> (b, n, h, v)
            # Transpose  -> (b, h, v, n)
            Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            # Let N = hh * ww,
            # Here rel_pos_emb.shape == (N, N, k, u)
            # rel_pos_emb[i][j] == a matrix represent the relationship between the i's pixel and j'pixel
            # Note that the image is already flattened here. So i and j are the indices on the flatten image
            rel_pos_emb = self.rel_pos_emb[n, m]
            # (m, u, n, k) * (m, u, b, v) => (m, u, b, v, n, k) (then m, u are summed up)
            # => (b, v, n, k) => (b, n, k, v) (for each pixel, there is a λp)
            λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b h v n', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out
