## Tensorflow implementation of NVAE:  A Deep Hierarchical Variational Autoencoder

My best shot at implementing the NVAE paper publish 8 July, 2020: https://arxiv.org/abs/2007.03898

### Reconstruction of 64x64 face images passing through NVAE
(I haven't gotten around yet to implementing the sampling to generate new faces)
![](images/nvae_roundtrip_samples.png)

This was created with a tiny version of the NVAE architecture
* num_scales=2
* num_groups_per_scale=2
* num_cells_per_group=2
* num_latent_per_group=20
* num_enc_channels=16
* num_prepost_blocks=1
* num_prepost_cells_per_block=2

![](images/small_nvae.png)


