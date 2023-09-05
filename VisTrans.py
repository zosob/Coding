import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ 
    Split  image into patches and then embed them.

    Parameters
    ----------

    img_size : in
        Size of the image (it is a square)

    patch_size : int
        Size of the patch (it is a square).

    in_chans : int
        Number of input channels.

    embed_dim : int
        The embedding dimension.

    
    Attributes
    ----------

    n_patched : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Conv layer that does both the splitting into
        patches and their embedding.    
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """Run forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape '(n_samples, in_chans, img_size, img_size, img_size)'.

        Returns
        -------

        torch.Tensor
            Shape `(n_smples, n_patches, embeded_dim)`.
        
        """

        x = self.proj(
            x
        )
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
    
class Attention(nn.Module):
    
   """ Attention mechanism.
        
        Parameters
        ----------
        dim : int
            The input and output dimension of per token features.

        n_heads : int
            Number of attention heads.

        qkv_bias : bool
            If true, then include bias to query, key and value tensor.

        proj_p : float
            Dropout probability applied to the outpud tensor.
        
        Attributes
        ----------

        scale : float
            Normalizing constant for the dot product.

        qkv : nn.Linear
            Linear projection for the query, key and value.

        proj : nn.Linear
            Linear mapping that takes in the concatenated output
            of all attention heads and maps it in to a new space.

        attn_drop, proj_drop : nn.Dropout
            Dropout layers.

        """     
   
   def __init__(self, dim, n_heads=12, qkv_bias = True, attn_p = 0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_p)
       
   def forward(self, x):
       """Run forward pass.
       
       Parameters
       ----------
       x: torch.tensor
        Shape `(n_smaples, n_patches+1, dim)`.

        Returns:
        -------
        x: torch.tensor
        Shape `(n_smaples, n_patches+1, dim)`.
    
       """
       n_samples, n_tokens, dim = x.shape
       if dim != self.dim:
           raise ValueError
       
       qkv = self.qkv(x)
       qkv = qkv.reshape(
           n_samples, n_tokens, 3, self.n_heads, self.head_dim
       )
       qkv = qkv.permute(
           2, 0, 3, 1, 4
       ) #(3, n_samples, n_heads, n_patches+1, head_dim)

       q, k, v = qkv[0], qkv[1], qkv[2]
       k_t = k.transpose(-2, -1)
       dp = (
           q @ k_t
       ) * self.scale

       attn = dp.softmac(dim=-1)
       attn = self.attn_drop(attn)

       weighted_avg = attn @ v
       weighted_avg = weighted_avg.transpose(
           1, 2
       )
       weighted_avg = weighted_avg.flatten(2) #concat attention head

       x = self.proj(weighted_avg)
       x = self.proj_drop(x)

       return x




