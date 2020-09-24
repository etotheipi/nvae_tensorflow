### (WIP) Tensorflow implementation of *NVAE:  A Deep Hierarchical Variational Autoencoder*

**WORK IN PROGRESS**

Implementation of the NVAE paper publish 8 July, 2020: https://arxiv.org/abs/2007.03898

Features and Limitations:
* ✓ Written in Tensorflow 2.3, Python 3.8.2
* ✓ Dynamic multi-scale, multi-group, multi-cell architecture ✓
* ✓ Spectral Normalization via tensorflow_addons ✓
* ✗ Residual normal distribution (not implemented) ✗ 
* ✗ Normalizing flows (not implemented) ✗ 


### Reconstruction of 64x64 face images passing through NVAE
![](images/nvae_roundtrip_samples.png)

### Generation of 64x64 face images 
![](images/nvae_gen_random.png)

This was created with a tiny version of the NVAE architecture
* num_scales=2
* num_groups_per_scale=2
* num_cells_per_group=2
* num_latent_per_group=20
* num_enc_channels=16
* num_prepost_blocks=1
* num_prepost_cells_per_block=2

![](images/small_nvae.png)


### Residual Cells

![](images/nvae_residual_cell_diagram.png)

