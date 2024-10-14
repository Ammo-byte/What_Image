Details of my project
# Image Captioning with CNN-RNN

This project is an **image captioning tool** that combines **Convolutional Neural Networks (CNNs)** with **Recurrent Neural Networks (RNNs)** to generate captions for images. The primary purpose of this project is to gain hands-on experience with **deep learning architectures**—specifically CNNs for feature extraction and RNNs for sequence generation. It serves as a practical introduction to how visual information can be converted into natural language.

Additionally, this is my **first deep learning project**, and it gave me **valuable first-hand experience** working with deep learning models. Through this project, I learned how to combine **visual and textual data**, preprocess datasets, and understand the inner workings of CNNs and RNNs. It provided me with a solid foundation for more advanced deep learning applications and gave insights into the challenges and rewards of working with neural networks.

---

## So What? Why is this Important?

This technology has practical applications in **assistive technologies**. For instance, this image captioning model could be embedded in **mobile apps or smart devices** to describe images to **visually impaired users**, allowing them to better understand their surroundings. It can generate captions describing objects, scenes, or activities detected in images, improving **accessibility** and enriching **user experience**.

---

## Understanding the Architecture (Image Explanation)

![Model Architecture](Model_Example_Diagram.png)

This image summarizes the **CNN-RNN architecture** used in the project:

1. **Input Image:**  
   The input image is resized to **224x224 pixels** and consists of **3 RGB channels**.

2. **Pretrained CNN Encoder:**  
   The CNN encoder (Inception v3) extracts **feature vectors** from the image. Pretrained on the **ImageNet dataset**, it identifies important image features (e.g., shapes, textures, and objects) to create a meaningful **feature vector** representation at the fully connected (fc) layer.

3. **Linear Transformation:**  
   The feature vector (size **1x1x2048**) is passed through a linear layer to adjust its shape to match the **embedding size** expected by the RNN decoder.

4. **LSTM Decoder:**  
   The RNN decoder, which consists of **LSTM (Long Short-Term Memory) layers**, generates a caption by predicting the next word in the sequence at each step. It starts with a `<start>` token and continues until it predicts the `<end>` token.

5. **Word Embeddings:**  
   Each word is represented as a **dense vector** through an embedding layer. The decoder generates a word at each step, and the process continues until the `<EOS>` token is predicted, marking the end of the caption.

---

## Codebase Overview

The project is broken down into several Python scripts. Below is a detailed explanation of each.

---

### 1. **`utils.py`** (Utility Functions)

This script contains **helper functions** for saving, loading, and printing captions during evaluation.

- **`print_examples()`**:  
  - This function tests the model by generating captions for sample images and prints both the correct and predicted captions.
  - It uses **transformations** to preprocess the images and **feeds them into the model** to generate captions.

- **`save_checkpoint()`**:  
  - Saves the model’s state and optimizer to a file so that training can be resumed later.

- **`load_checkpoint()`**:  
  - Loads a previously saved checkpoint, restoring the model and optimizer state.

---

### 2. **`inference.py`** (Caption Generation)

This script demonstrates how to **load a pretrained model** and generate captions for **new images**.

- **Vocab loading with `pickle`**:  
  - Loads the vocabulary used during training from `vocabulary.pkl`.

- **Model Loading**:  
  - Loads the saved checkpoint from `checkpoint_epoch_20.pth.tar` and restores the model state.

- **Image Preprocessing**:  
  - Resizes the input image to **299x299** and normalizes it before passing it to the model.

- **Caption Generation**:  
  - Uses the `image_caption()` function to generate and print captions for the provided image.

---

### 3. **`image_training.py`** (Model Training)

This script defines the **training loop** for the image captioning model. Below are the key steps:

- **Data Loading and Preprocessing**:  
  - Loads the dataset using a custom DataLoader from the `image_loader.py` script. Each image is resized, normalized, and converted into tensors.

- **Hyperparameters Setup**:  
  - Defines key parameters like **embedding size**, **hidden size**, **number of layers**, **learning rate**, and **number of epochs**.

- **Model Setup**:  
  - Uses the `CNN_RNN` class to create the combined model. The CNN extracts image features, and the RNN generates captions.

- **Training Loop**:  
  - For each epoch, the model processes batches of images and captions. The loss is calculated using **CrossEntropyLoss**, and the optimizer updates the model weights.

- **Checkpoint Saving**:  
  - Every 5 epochs, the model’s state is saved to disk, allowing for resumption from that point if needed.

---

### 4. **`image_intial_model.py`** (Model Definition)

This script contains the **core architecture** of the model, including the CNN encoder and the RNN decoder.

- **`CNN_encoder`**:  
  - Uses a pretrained **Inception v3 model** from `torchvision` to extract feature vectors from input images.
  - The final **fully connected (fc)** layer of the Inception model is replaced to produce features matching the **embedding size** required by the decoder.

- **`RNN_decoder`**:  
  - The decoder consists of **LSTM layers** to handle sequential word generation. It takes the feature vector from the encoder along with the current word embedding to predict the next word.

- **`CNN_RNN` Class**:  
  - Combines the CNN encoder and the RNN decoder into a single end-to-end model.
  - The **`image_caption()`** function generates captions for input images by extracting features with the encoder and generating sequences with the decoder.

---

### 5. **`image_loader.py`** (Data Loader)

This script handles **loading and preparing the dataset** for training.

- **`Vocab` Class**:  
  - Builds the **vocabulary** from the captions by tokenizing each caption and assigning a numerical index to each word.
  - Provides functions to **convert words to indices** and **numeralize captions**.

- **`FlickrDataset` Class**:  
  - Loads images and captions from disk and prepares them for training.
  - Handles image preprocessing using the **Pillow** library and transforms them into tensors.

- **Collate Function**:  
  - Pads batches of captions to ensure they are of uniform length when passed to the model during training.

---

## Requirements

To run the project, install the following dependencies:

```bash
pip install torch torchvision pandas spacy tqdm pillow
python -m spacy download en_core_web_sm
```

You will also need:

- Python 3.x  
- **[Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)** (download from Kaggle)  
- A **GPU** for faster training (optional)

---

## Shoutout and Inspiration

A big **shoutout to [Aladdin Persson](https://www.youtube.com/c/AladdinPersson)**, whose tutorials were instrumental in building this project. His YouTube channel offers clear and practical deep learning explanations, and I highly recommend checking it out if you’re interested in **computer vision and NLP applications**.

This project draws inspiration from his video on **image captioning** and serves as a hands-on way to **explore CNNs and RNNs** in real-world tasks.

---

## Conclusion

This project provides a **practical exploration of deep learning**, demonstrating how CNNs and RNNs work together to generate image captions. The technology has immense potential in **assistive applications** for the visually impaired, helping them understand visual content through textual descriptions. It also highlights the importance of **multimodal learning**, where visual and textual information are combined to achieve a task.

By working through this project, you’ll gain experience in:

- **Feature extraction using CNNs**  
- **Sequence modeling with RNNs (LSTMs)**  
- **Training deep learning models** with PyTorch

Feel free to explore and extend this project as you wish. If you found this helpful, consider giving it a ⭐ on GitHub!

---

## Acknowledgments

- **Flickr8k Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/adityajn105/flickr8k)  
- **Inception v3**: Pretrained model from **Torchvision**  
- **TensorBoard**: For tracking and visualizing the training process  
