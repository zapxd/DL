# Deep Learning - Complete Assignment Notes

## Table of Contents

- [Assignment 1: Deep Learning Packages](#assignment-1-deep-learning-packages)
- [Assignment 2: Feed Forward Neural Networks](#assignment-2-feed-forward-neural-networks)
- [Assignment 3: Convolutional Neural Networks (CNN)](#assignment-3-convolutional-neural-networks-cnn)
- [Assignment 4: Anomaly Detection using Autoencoders](#assignment-4-anomaly-detection-using-autoencoders)
- [Assignment 5: Natural Language Processing - CBOW Model](#assignment-5-natural-language-processing---cbow-model)
- [Assignment 6: Transfer Learning and Object Detection](#assignment-6-transfer-learning-and-object-detection)

---

## Assignment 1: Deep Learning Packages

### What is Deep Learning?
Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers to automatically learn representations from data. It is inspired by the structure and function of the human brain and is mainly used for tasks like image recognition, natural language processing, and speech recognition.

### Python Packages for Machine Learning and Deep Learning

**Machine Learning Libraries:**
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

**Deep Learning Libraries:**
- TensorFlow
- Keras
- Theano
- PyTorch

### Framework Comparison

| Criteria | TensorFlow | Keras | Theano | PyTorch |
|-----------|-------------|--------|---------|----------|
| **Type** | Full-fledged framework | High-level API | Mathematical computations | Dynamic computation framework |
| **GUI** | TensorBoard | Integrated with TensorBoard | No GUI | TensorBoard support |
| **Features** | Scalable, distributed training | User-friendly, easy prototyping | Symbolic computation | Dynamic graphs, intuitive debugging |
| **Compatibility** | Python, C++, JavaScript, Java | Runs on TensorFlow, CNTK, Theano | NumPy and SciPy | Python, C++, CUDA |
| **Applications** | Large-scale ML, NLP, Computer Vision | Fast prototyping | Academic research | Computer Vision, NLP, reinforcement learning |

### TensorFlow Ecosystem

**Models:**
- MobileNet, ResNet, Inception, EfficientNet, BERT

**Datasets:**
- MNIST, CIFAR-10, ImageNet, COCO

**Tools and Extensions:**
- **TensorFlow Lite:** For mobile devices
- **TensorFlow.js:** For web applications
- **TensorFlow Extended (TFX):** Production pipelines
- **TensorBoard:** Visualization tool
- **TensorFlow Hub:** Pretrained models repository
- **TensorFlow Serving:** Model deployment

### Keras Ecosystem

**Components:**
- **KerasTuner:** Hyperparameter tuning automation
- **KerasNLP:** Natural Language Processing tools (BERT, GPT)
- **KerasCV:** Computer vision models (object detection, segmentation)
- **AutoKeras:** Automatic model building and optimization
- **Model Optimization Toolkit:** Pruning, quantization, performance tuning

### Theano
- One of the first deep learning frameworks
- Introduced symbolic computation and GPU support
- Uses symbolic variables and compiles computation graphs into efficient C code
- Laid foundation for TensorFlow and Keras
- Last stable version: 1.0.5

### PyTorch

**Key Features:**
- **Tensors:** Multi-dimensional arrays with GPU acceleration
- **Dynamic Computation Graphs:** Graphs built on-the-fly during execution
- **Automatic Differentiation:** Through `torch.autograd`
- **GPU Support:** CUDA integration for faster computation

**Related Technologies:**
- **Uber's Pyro:** Deep probabilistic programming framework
- **Tesla Autopilot:** Uses PyTorch for autonomous driving perception

### Virtual Environments
Virtual environments allow isolation of dependencies for different projects, preventing version conflicts between libraries. Each environment maintains its own Python binaries and package versions.

### Why Upgrade pip?
Upgrading pip ensures compatibility with latest package versions and prevents installation errors caused by outdated dependencies.

---

## Assignment 2: Feed Forward Neural Networks

### What is a Feed Forward Neural Network (FNN)?
A Feed Forward Neural Network is a type of artificial neural network where connections between nodes do not form cycles. Information flows in one direction: from input layer → through hidden layers → to output layer.

### MNIST Dataset
- **Modified National Institute of Standards and Technology dataset**
- Contains 70,000 handwritten digits (0-9)
- Split: 60,000 training images + 10,000 test images
- Image size: 28×28 pixels (grayscale)
- Commonly used for image classification benchmarks

### TensorFlow
An open-source machine learning and deep learning framework developed by Google. It allows creation of computational graphs for building and training neural networks.

### Keras
A high-level deep learning API running on top of TensorFlow. Simplifies neural network development through easy-to-use abstractions like Sequential models and layers.

### Tensor
A multidimensional array used to represent data in TensorFlow. Can be:
- Scalar (0D)
- Vector (1D)
- Matrix (2D)
- n-dimensional array

### Neural Network Layers

#### Flatten Layer
- Converts multi-dimensional input into a single vector
- Reshapes 2D 28×28 images into 1D arrays (784 elements)
- Prepares data for dense layers without changing actual data

#### Dense Layer (Fully Connected Layer)
- Connects each neuron in previous layer to every neuron in current layer
- Computes weighted sum of inputs + bias
- Passes result through activation function

### Activation Functions

**Purpose:** Introduce non-linearity to learn complex patterns

**Common Types:**
- **ReLU (Rectified Linear Unit):**
  - Sets negative values to zero
  - Speeds up convergence
  - Avoids vanishing gradient problem
- **Softmax:**
  - Used in output layer for multi-class classification
  - Converts outputs to probability distribution (sum = 1)

### Training Components

#### Optimizer
Algorithm to minimize loss by updating weights. **SGD (Stochastic Gradient Descent)** updates weights based on data batches for faster convergence.

#### Loss Function
**Sparse Categorical Crossentropy** - suitable for multi-class classification where labels are integers (not one-hot encoded).

#### Metrics
Performance measures like **Accuracy** to evaluate model.

### Training Concepts

**Epoch:** One complete pass of entire training dataset through the network.

**model.compile():** Defines optimizer, loss function, and metrics.

**model.fit():** Trains the network for specified epochs, returns history object with loss/accuracy details.

**model.evaluate():** Calculates loss and accuracy on test dataset.

**model.predict():** Generates predictions (probability values for each class).

### Accuracy Types

- **Training Accuracy:** Performance on training data
- **Validation Accuracy:** Performance on unseen data during training
- Large difference indicates **overfitting**

### Overfitting
Model performs well on training data but poorly on unseen data. Means model memorized training data instead of learning general patterns.

**Prevention Techniques:**
- Dropout layers
- Data augmentation
- Early stopping
- Regularization (L1/L2)
- Batch normalization

### Normalization
Scaling pixel values from 0-255 range to 0-1 range improves training stability and convergence speed.

### Learning Rate Effects
- **Too High:** Model overshoots minima, fails to converge
- **Too Low:** Very slow convergence, may get stuck in local minima

### Confusion Matrix
A table visualizing model performance by showing correct and incorrect classifications for each class. Helps identify where model makes errors.

---

## Assignment 3: Convolutional Neural Networks (CNN)

### What is Image Classification?
The process of categorizing and labeling images into specific classes based on visual features using machine learning or deep learning techniques.

### What is a CNN?
A Convolutional Neural Network is a deep learning model designed for processing image data. It automatically detects spatial patterns like edges, textures, and shapes through convolutional and pooling operations.

### Why CNNs over Traditional Neural Networks?
CNNs automatically extract spatial features using:
- Local connections
- Shared weights
- Hierarchical feature learning

Traditional MLPs treat all pixels independently and require manual feature extraction.

### CNN Architecture Layers

#### 1. Convolutional Layer
- Applies filters (kernels) over images
- Extracts local features: edges, textures, corners
- Creates feature maps capturing spatial relationships

#### 2. ReLU Activation Layer
- Introduces non-linearity
- Replaces negative values with zero
- Improves training speed and convergence

#### 3. Pooling Layer
- Reduces feature map dimensions
- Keeps most important information
- Controls overfitting
- Reduces computation

**Types:**
- Max Pooling: Takes maximum value
- Average Pooling: Takes average value

#### 4. Flatten Layer
- Converts 2D feature maps to 1D vector
- Prepares data for fully connected layers

#### 5. Dense (Fully Connected) Layer
- Performs final classification
- Learns complex combinations of features

#### 6. Softmax Output Layer
- Converts outputs to probability distribution
- Probabilities sum to 1
- Determines most likely class

### Model Performance Indicators

**High Accuracy + Low Loss indicates:**
- Model learned well
- Makes confident predictions
- Generalizes effectively to unseen data

### Model Generalization
Model's ability to perform well on unseen data, not just training data.

### Real-World CNN Applications
- Face and object recognition
- Medical image analysis
- Self-driving cars
- Security and surveillance
- Optical character recognition (OCR)

---

## Assignment 4: Anomaly Detection using Autoencoders

### What is an Autoencoder?
An unsupervised neural network that learns to reconstruct its input. Compresses data into lower-dimensional **latent representation** then reconstructs it back using a **decoder**.

### Autoencoder Structure

**Three Main Components:**

1. **Encoder**
   - Compresses input data
   - Creates latent space representation
   - Direction: Input → Latent

2. **Latent Space (Bottleneck Layer)**
   - Compressed feature representation
   - Captures essential patterns
   - Most compact information form

3. **Decoder**
   - Reconstructs from latent representation
   - Direction: Latent → Output
   - Recreates original input

### Anomaly Detection Process

1. Train Autoencoder on **normal data only**
2. Use trained model to reconstruct new data
3. Calculate **reconstruction error**
4. Set threshold for error
5. Points above threshold = **Anomalies**

### Why Autoencoders for Anomaly Detection?
- Learns patterns of normal data
- Anomalies produce high reconstruction errors
- Deviations from learned patterns are easily detected

### Learning Type
**Unsupervised Learning** - No labeled data required. Network learns to map input to itself.

### Latent Representation
Compressed form of input data capturing essential features and structure in fewer dimensions.

### Reconstruction Error
The difference between original input and reconstructed output. Quantifies how well the Autoencoder reproduces input data.

**Calculation:** Mean Squared Error (MSE) between input and output

### Anomaly Detection Logic
```
If reconstruction_error > threshold:
    Label = Anomaly
Else:
    Label = Normal
```

### Dataset Information
**ECG Dataset** used where:
- Label 1 = Normal observations
- Label 0 = Anomalies

### Training Strategy
**Why train only on normal data?**
- Model needs to learn normal behavior patterns
- When encountering anomalies, reconstruction fails
- Results in higher reconstruction error for anomalies

### Preprocessing
**Normalization/Scaling** using StandardScaler:
- Ensures all features on similar scale
- Stabilizes training
- Speeds up convergence
- Improves performance

### Training Components

**Optimizer:** Adam or SGD (adjusts weights during training)

**Loss Function:** Mean Squared Error (MSE) - measures difference between input and reconstructed output

### High Reconstruction Error Indicates
Data point doesn't follow learned normal pattern → likely an **Anomaly**

### Advantages

- Effective for **imbalanced datasets** (abundant normal data, rare anomalies)
- Automatic complex feature learning
- No manual feature engineering required
- Handles **high-dimensional data** efficiently

### Limitations

- Requires large amounts of normal data
- Sensitive to hyperparameters and architecture
- May not generalize if normal data too diverse

### Threshold Selection Methods

- Statistical: mean + 3×standard deviation
- Visual inspection of error distributions
- ROC curve analysis
- Percentile-based approaches

### Imbalanced Dataset
Dataset where one class heavily outnumbers the other (e.g., 99% normal, 1% anomalies). Makes traditional classification methods ineffective.

### Evaluation Metrics

- **Precision:** Proportion of true anomalies among predicted anomalies
- **Recall:** Proportion of actual anomalies correctly identified
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve

### Why High Accuracy Can Be Misleading
In imbalanced datasets, model can achieve high accuracy by predicting everything as normal class, but fails to detect anomalies (low recall).

### Training with Both Normal and Anomalous Data
**Problem:** Model learns to reconstruct anomalies too, reducing reconstruction error differences. Makes anomaly detection ineffective.

### Real-World Applications

- Credit card fraud detection
- Network intrusion detection
- Industrial equipment fault detection
- Medical anomaly detection (ECG, MRI)
- Cybersecurity and fraud analytics

---

## Assignment 5: Natural Language Processing - CBOW Model

### What is NLP?
Natural Language Processing is a field of Artificial Intelligence enabling computers to understand, interpret, and generate human language. Combines linguistics, computer science, and machine learning.

### Word Embedding
Technique to represent words as dense numerical vectors in continuous vector space. Words with similar meanings have similar vector representations, capturing semantic relationships.

### Word2Vec Techniques

**Two Main Architectures:**

1. **CBOW (Continuous Bag of Words)**
   - Predicts target word from surrounding context words
   - Faster, efficient for large datasets
   - Better for frequent words

2. **Skip-Gram**
   - Predicts context words from target word
   - Slower, works better with small datasets
   - Better for rare words

### Word Embedding Applications

- Sentiment analysis
- Machine translation
- Text classification
- Chatbots and conversational AI
- Document similarity detection
- Named entity recognition (NER)

### CBOW Architecture

**Structure:**

1. **Input Layer**
   - Takes multiple context words as input

2. **Projection/Embedding Layer**
   - Converts each context word to embedding vector

3. **Hidden Layer**
   - Averages context embeddings
   - Forms context representation

4. **Output Layer**
   - Predicts target word based on representation

### CBOW Input and Output

**Example:** "The cat sits on the mat"

**If target word = "sits" and window_size = 2:**
- **Input:** ["The", "cat", "on", "the"]
- **Output:** ["sits"]

### Tokenizer
Preprocessing tool that splits text into individual tokens (words or subwords). Converts sentences into numerical data suitable for machine learning models.

### Window Size Parameter

Determines how many words on each side of target word used as context.

**Example:** Window size = 2 in "The cat sits on the mat"
- Target: "sits"
- Context: ["The", "cat", "on", "the"]

**Effects:**
- **Larger window:** Captures more global context
- **Smaller window:** Captures local context

### Keras Layers for NLP

#### Embedding Layer
- Converts integer-encoded words into dense vectors of fixed size
- Learns word representations during training
- Example: Converts word indices into 100-dimensional vectors

#### Lambda Layer
- Performs custom operations within neural network
- Used for averaging embeddings or mathematical operations
- No trainable parameters

### Python yield() Keyword
Used in generators to produce sequence of results lazily (one at a time). Returns data iteratively instead of computing everything at once.

**Benefits:**
- Saves memory
- Improves performance
- Efficient for data generation in batches

### CBOW vs Skip-Gram Comparison

| Feature | CBOW | Skip-Gram |
|----------|-------|-----------|
| **Objective** | Predicts target from context | Predicts context from target |
| **Speed** | Faster | Slower |
| **Dataset Size** | Better for large datasets | Better for small datasets |
| **Performance** | Better for frequent words | Better for rare words |

### Gensim Library
Python library for topic modeling and word embedding. Provides implementations of Word2Vec, Doc2Vec, and other models for text representation and semantic analysis.

### Word Vectors Importance
Help machines understand word meaning and relationships by mapping them into vector space where similar words are positioned closer together.

**Example:** king - man + woman ≈ queen

### One-Hot Encoding in NLP

**Definition:** Represents words as binary vectors where:
- One position (word index) = 1
- All other positions = 0

**Purpose:** Uniquely identifies words before converting to dense embeddings

**Limitation:** 
- Doesn't capture semantic relationships
- Results in sparse, high-dimensional vectors
- Word embeddings are dense and capture contextual meaning

### CBOW Output Representation
Represents predicted **probability distribution** of words in vocabulary. Word with highest probability is most likely target word given the context.

### CBOW Hyperparameters

- Window size
- Embedding dimension
- Learning rate
- Number of epochs
- Optimizer type (Adam, SGD)

### Finding Similar Words
After training, use **Gensim KeyedVectors** class and `most_similar()` method to find words with similar vector representations.

---

## Assignment 6: Transfer Learning and Object Detection

### What is Transfer Learning?
Technique where a model trained on one large dataset is reused (fully or partially) for another related task. Instead of training from scratch, use a **pretrained model** and fine-tune it.

**Benefits:**
- Saves time and computational resources
- Requires less data
- Leverages previously learned features

### Pretrained Neural Network Models
Deep learning models already trained on large benchmark datasets (like ImageNet). Have learned to extract useful features:
- Edges
- Textures
- Shapes
- Complex patterns

**Examples:** VGG16, ResNet, Inception, MobileNet, EfficientNet

### PyTorch Library
Open-source deep learning framework developed by Facebook. Provides flexible tools for building and training neural networks.

**Key Features:**
- Dynamic computation graphs
- GPU acceleration
- Pythonic and intuitive API

### Transfer Learning Advantages

- Reduces training time significantly
- Requires less training data
- Improves performance on small datasets
- Helps models generalize better
- Leverages previously learned features
- Cost-effective

### Transfer Learning Applications

- Image classification and object detection
- Face recognition
- Medical image analysis
- Natural language processing (NLP)
- Speech recognition
- Autonomous vehicles

### Important Datasets

#### Caltech-101 Dataset
- Contains ~9,000 labeled images
- Divided into 101 object categories
- Each class: 40-800 images
- Varied image sizes
- Used for object recognition and classification

#### ImageNet Dataset
- One of largest image databases
- Over 14 million labeled images
- 1,000 object categories
- Widely used for training and evaluating deep learning models
- Standard benchmark for image classification

### Transfer Learning Steps

1. **Load pretrained CNN model** (e.g., VGG16, ResNet)
2. **Freeze lower convolutional layers** (retain general features)
3. **Add custom classifier** on top with trainable layers
4. **Train classifier** on specific dataset
5. **Fine-tune model** by unfreezing more layers with smaller learning rate

### Data Augmentation

**Definition:** Technique to artificially expand training dataset by applying random transformations.

**Common Transformations:**
- Rotation
- Flipping
- Cropping
- Scaling
- Color jittering

**Benefits:**
- Helps model generalize better
- Reduces overfitting
- Simulates data diversity
- Important when working with small datasets

### Preprocessing in Transfer Learning

**Purpose:**
- Ensures input images match pretrained model expectations
- Maintains compatibility with pretrained weights
- Improves model performance

**Requirements:**
- Resize to expected dimensions (e.g., 224×224)
- Normalize pixel values
- Apply same transformations as original training

### PyTorch Transforms Module

**torchvision.transforms** - provides image transformation operations

#### Common Training Transforms

- **RandomResizedCrop(size, scale):** Random crop with scaling
- **RandomRotation(degrees):** Rotates within specified degrees
- **ColorJitter():** Changes brightness, contrast, saturation
- **RandomHorizontalFlip():** Horizontal image flip
- **CenterCrop(size):** Crops center region
- **ToTensor():** Converts to PyTorch tensor
- **Normalize(mean, std):** Normalizes using statistics

#### Validation Transforms
Simpler than training transforms (no random alterations):
- Resize to fixed size (e.g., 256×256)
- Center crop to 224×224
- Convert to tensor
- Normalize using ImageNet statistics

**Purpose:** Ensure consistency across validation epochs

### VGG-16 Model Architecture

**Developed by:** Visual Geometry Group at Oxford

**Structure:**
- 13 convolutional layers
- 3 fully connected layers
- 138 million parameters
- Input: 224×224×3 images

**Key Characteristics:**
- Uses small 3×3 filters throughout
- Simple and uniform architecture
- Commonly used for transfer learning

### Freezing Layers

**Definition:** Stopping layer weights from updating during backpropagation

**Purpose:**
- Retains pretrained low-level features (edges, textures)
- Only trains new classifier layers for specific task
- Faster training
- Prevents catastrophic forgetting

### Fine-Tuning

**Process:**
- Unfreeze some higher layers of pretrained network
- Train them alongside new layers
- Use smaller learning rate
- Allows model to adapt to new dataset
- Preserves learned representations

### Dropout Layer

**Regularization technique:**
- Randomly deactivates fraction of neurons during training
- Prevents overfitting
- Ensures model doesn't rely too much on specific neurons
- Improves generalization

### LogSoftmax Function

**Purpose:**
- Converts final output logits to log probabilities
- Used with Negative Log Likelihood Loss (NLLLoss)
- For multi-class classification in PyTorch

### Training Components

**Loss Function:** Negative Log Likelihood Loss (NLLLoss)
- Measures how well predicted probability distribution matches true labels

**Optimizer:** Adam optimizer
- Adjusts model parameters efficiently based on gradients
- Combines advantages of AdaGrad and RMSProp

### Early Stopping

**Definition:** Halts training when validation loss stops improving for several epochs

**Benefits:**
- Prevents overfitting
- Saves computational resources
- Finds optimal point before model starts memorizing

### Training vs Validation Accuracy

**Training Accuracy:**
- Measures performance on training data

**Validation Accuracy:**
- Evaluates generalization on unseen data

**Large gap between them = Overfitting**

### ImageNet Normalization Values

**Mean:** [0.485, 0.456, 0.406]
**Std Dev:** [0.229, 0.224, 0.225]

**Purpose:**
- Represent pixel intensity distributions of ImageNet
- Using same normalization ensures consistency
- Maintains compatibility with pretrained model expectations

### Benefits of Pretrained Models

- Already learned general visual features
- Significantly reduce training time
- Reduce data requirements
- Provide strong performance on small datasets
- State-of-the-art feature extractors

---

## Key Concepts Summary

### Neural Network Fundamentals

**Activation Functions:** Introduce non-linearity for learning complex patterns

**Optimization:** Process of adjusting weights to minimize loss

**Backpropagation:** Algorithm for computing gradients and updating weights

**Loss Functions:** Measure difference between predictions and actual values

### Model Training Concepts

**Overfitting:** Model memorizes training data, poor generalization

**Underfitting:** Model too simple to capture patterns

**Generalization:** Ability to perform well on unseen data

**Regularization:** Techniques to prevent overfitting

### Performance Evaluation

**Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Validation:** Testing model on data not used during training

**Cross-Validation:** Multiple validation splits for robust evaluation

---

**Note:** This document covers theoretical concepts and definitions from all assignments. For practical implementation, refer to the code examples in your lab assignments.
