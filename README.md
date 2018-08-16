# MNIST experiments
Implementation of three different Convolutional Neural Network (CNN) architectures benchmarked on the MNIST dataset. Evaluation on the quite simple dataset is mostly due to lack of powerful GPUs. :D
* VGGNet: The 2014 runners-up spot of the ImageNet Challenge is characterized by it's 3 layer blocks: 3x3 conv, 3x3 conv, 2x2 max_pool; which are very effective and save parameters in comparison with AlexNets 11x11 conv and 5x5 conv 
* GoogLeNet (InceptionV1): The 2014 winner of the ImageNet Challenge introduced the concept of inception to CNNs, which parallelizes ConvNets to reach a better gradient propagation through deep nets
* ResNet: The 2015 winner of the ImageNet Challenge introduced the concept of residual connections to CNNs, making the gradient propagation to the higher layers even easier than with inception. With this architecture, CNNs with up to 1000 layers are possible.

The characteristic modules described above are implemented for each of my CNN architectures, however their are not as deep as their originals, since I needed to save parameters due to GPU limitations and of course the simpler dataset I used for evaluation which does not demand so many parameters.

Evalution Results (accuracy):
1. VGGNet-like: 99.63%
1. ResNet-like: 99.61%
1. GoogLeNet-like: 99.42%

With more hyperparameter tuning one might get better results since those are not really state-of-the-art, which would be around 99.8%. However, I did this only for exercise so I'm fine with the results.

Conclusion: On simpler architectures and datasets, neither the inception module nor the residual connections can really add benefit.