# Image_colorization_using_deep_transfer_learning
### Model output:

![t1](https://github.com/urvilatnekar/Image_colorization_using_deep_transfer_learning/assets/30819058/5523d2ab-f3d6-402d-b39f-551315a0f0b5)
![0](https://github.com/urvilatnekar/Image_colorization_using_deep_transfer_learning/assets/30819058/e463b7a6-684d-4c52-bb1e-e2ad60dce57c)

### Example 1

![t3](https://github.com/urvilatnekar/Image_colorization_using_deep_transfer_learning/assets/30819058/bd6afb58-08bf-4373-bba4-e47df9308039)
![2](https://github.com/urvilatnekar/Image_colorization_using_deep_transfer_learning/assets/30819058/795d6d9a-27f4-4480-ba23-be67d1d00e7a)

### Example 2

![t2](https://github.com/urvilatnekar/Image_colorization_using_deep_transfer_learning/assets/30819058/50bda526-d402-4698-a032-655b2df92b09)
![1](https://github.com/urvilatnekar/Image_colorization_using_deep_transfer_learning/assets/30819058/dbaf9a4a-b199-4685-b8fa-6712288c8fa3)

### Example 3

Image colorization is the process of adding color to grayscale or black-and-white images to create visually appealing and realistic color versions of the original images. It involves using computer algorithms and techniques to automatically assign appropriate colors to different parts of an image based on various factors such as context, content, and color references.
Autoencoders are a type of neural network architecture that is primarily used for unsupervised learning tasks, particularly in the field of deep learning. The main objective of an autoencoder is to learn a compressed representation, or encoding, of input data, and then reconstruct the original input from this encoded representation. The encoder part takes the input data and transforms it into a lower-dimensional representation, capturing the most important features or patterns of the input. The decoder part then takes this encoded representation and reconstructs the original input as closely as possible.
Autoencoders have several applications and benefits:
1)Dimensionality Reduction 2)Anomaly Detection 3)Data Denoising 4)Feature Learning 5)Generative Modeling

The architecture of autoencoders consists of two main components: an encoder and a decoder. It follows an unsupervised learning approach and aims to learn an efficient representation (encoding) of the input data and reconstruct the original input data (decoding) as accurately as possible.
1.	Encoder: The encoder takes the input data and maps it to a lower-dimensional representation, also known as the latent space or bottleneck layer. The encoder network consists of multiple layers, typically implemented as fully connected (dense) layers or convolutional layers in the case of image data. Each layer applies a non-linear activation function, such as ReLU (Rectified Linear Unit), to introduce non-linearity into the encoding process. The number of nodes in the bottleneck layer is usually smaller than the dimensionality of the input data, leading to dimensionality reduction.
2.	Latent Space: The bottleneck layer represents a compressed and encoded version of the input data. It captures the most important features and patterns in a lower-dimensional space. The dimensionality of the latent space is a design choice and depends on the complexity of the data and the desired level of compression.
3.	Decoder: The decoder takes the encoded representation from the bottleneck layer and reconstructs the original input data. Similar to the encoder, the decoder network consists of multiple layers with non-linear activation functions. The final layer of the decoder typically uses an activation function appropriate for the type of data being reconstructed, such as sigmoid for binary data or softmax for multi-class data. The output layer aims to match the dimensions of the original input data

Transfer learning is a machine learning technique where a model trained on one task or domain is reused or adapted to perform a different but related task or operate in a different domain. In transfer learning, the pretrained model serves as a starting point or a feature extractor, capturing generic knowledge and patterns from the source task or domain. 

In the context of deep learning and computer vision, the term "backbone" refers to the main or primary architecture of a neural network model that is responsible for extracting meaningful features from the input data. The choice of backbone architecture depends on the specific task or problem at hand. Popular backbone architectures in computer vision include VGGNet, ResNet, InceptionNet, and MobileNet, among others. These architectures are pre-trained on large-scale datasets like ImageNet and serve as effective feature extractors for various computer vision tasks, such as image classification, object detection, and segmentation.

VGG16, short for Visual Geometry Group 16, is a convolutional neural network (CNN) architecture and VGG16 was originally trained on the ImageNet dataset, which consists of millions of labeled images spanning 1,000 different classes. It achieved state-of-the-art performance on the image classification task of the ILSVRC, demonstrating the effectiveness of deep CNNs in visual recognition tasks.
Here are the general steps to train only the upper layers using VGG for transfer learning:
1.	Load the pre-trained VGG model without the fully connected layers:
•	Remove the top (fully connected) layers of the VGG model.
•	Ensure that the weights of the VGG layers are set to non-trainable.
2.	Define and add your own custom decoder layers on top of the VGG model:
•	These layers are responsible for transforming the extracted features into the desired output format or performing the specific task you're working on (e.g., image colorization).
3.	Compile the model:
•	Specify the loss function, optimizer, and any other metrics you want to track during training.
4.	Load and preprocess your dataset:
•	Prepare your dataset according to the input requirements of the VGG model.
5.	Train the model:
•	Pass your training data through the VGG model (which acts as a feature extractor) to obtain the extracted features.
•	Feed the extracted features into the custom decoder layers and train them using the desired loss function.
•	Update only the weights of the decoder layers while keeping the VGG layers frozen.
6.	Evaluate and fine-tune:
•	After training the upper layers, evaluate the performance of your model on a validation set.
•	If necessary, you can further fine-tune the model by adjusting hyperparameters, modifying the architecture, or considering unfreezing and training some of the VGG layers.

