# Deep_image_prior_inpainting

Implementation of the Deep Image Prior paper for a school project for inpainting with Edge Connect GAN for comparison

# Official implementation

https://github.com/DmitryUlyanov/deep-image-prior

# About the code

Inside the model folder, the different classes used to build the models can be found. The EdgeModelInpainting is directly taken from the EdgeConnect paper. Other models have been implemented reproducing Encoder-Decoder CNN models described in the Deep Image Prior paper. The models focusing on Residual Connections and Skip Connections make respectively use of element-wise addition and concatenation between different layers.

# Reproducibility

To reproduce the results, you can use the jupyter notebook associated : Deep Image Prior. Inside you will be able to chose which particular model you want and on which image. To add more images, you can add an image file in the data folder, we will also have to add a mask corresponding to that image. The mask should be a grayscale image (with only one channel), it is preferable to use jpg images.

# Citation

```
@article{UlyanovVL17,
    author    = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
    title     = {Deep Image Prior},
    journal   = {arXiv:1711.10925},
    year      = {2017}
}
```

```
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}
```

```
@InProceedings{Nazeri_2019_ICCV,
  title = {EdgeConnect: Structure Guided Image Inpainting using Edge Prediction},
  author = {Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
  month = {Oct},
  year = {2019}
}
```