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