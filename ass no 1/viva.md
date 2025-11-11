# Viva Questions and Answers  
## Assignment No.1  
### Title: Study of Deep Learning Packages – TensorFlow, Keras, Theano, and PyTorch  

---

### **1. What is Deep Learning?**
Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers to automatically learn representations from data. It is inspired by the structure and function of the human brain and is mainly used for tasks like image recognition, natural language processing, and speech recognition.

---

### **2. What are various Python packages supporting Machine Learning libraries, and which are mainly used for Deep Learning?**
Python provides several libraries for Machine Learning and Deep Learning:
- **Machine Learning:** Scikit-learn, NumPy, Pandas, Matplotlib.
- **Deep Learning:** TensorFlow, Keras, Theano, PyTorch.

TensorFlow, Keras, Theano, and PyTorch are the most widely used packages for building and training deep learning models.

---

### **3. Comparison: TensorFlow / Keras / Theano / PyTorch**

| Criteria | TensorFlow | Keras | Theano | PyTorch |
|-----------|-------------|--------|---------|----------|
| **Available Functionality** | Full-fledged framework for building, training, and deploying ML/DL models | High-level API running on top of TensorFlow | Focuses on mathematical computations and symbolic differentiation | Dynamic computation graph and tensor-based operations |
| **GUI Status** | TensorBoard for visualization | Integrated with TensorBoard | No GUI | TensorBoard support and third-party visualizers |
| **Versions** | TensorFlow 2.x (latest) | Keras 3.x (integrated with TF 2.x) | Last stable version 1.0.5 | PyTorch 2.x (latest) |
| **Features** | Scalable, supports distributed training, good for production | User-friendly, high-level API, easy prototyping | Symbolic computation, fast numerical operations | Dynamic graphs, intuitive debugging, GPU support |
| **Compatibility** | Works with Python, C++, JavaScript, and Java | Runs on top of TensorFlow, CNTK, or Theano | Works with NumPy and SciPy | Works with Python, C++, CUDA |
| **Specific Applications** | Large-scale ML, NLP, Computer Vision | Fast prototyping and model design | Academic research and numerical computations | Computer Vision, NLP, reinforcement learning |

---

### **4. TensorFlow Models, Datasets, and Tools**

**Models and Pretrained Models:**
- MobileNet, ResNet, Inception, EfficientNet, BERT.

**Datasets:**
- MNIST, CIFAR-10, ImageNet, COCO.

**Libraries and Extensions:**
- TensorFlow Lite (for mobile),
- TensorFlow.js (for web),
- TensorFlow Extended (TFX) for production pipelines,
- TensorBoard for visualization.

**Tools:**
- TensorFlow Hub for pretrained models,
- TensorFlow Model Garden for research models,
- TensorFlow Serving for deployment.

**Case Studies:**
- **PayPal:** Uses TensorFlow for detecting frauds and analyzing customer transactions.
- **Intel:** Uses TensorFlow for optimizing hardware acceleration and AI performance in Intel processors.

---

### **5. Explain the Keras Ecosystem**

**Keras Ecosystem includes:**
- **KerasTuner:** Automates hyperparameter tuning.
- **KerasNLP:** Provides tools for building NLP models like BERT, GPT, etc.
- **KerasCV:** For computer vision models such as object detection and segmentation.
- **AutoKeras:** For automatic model building and optimization.
- **Model Optimization Toolkit:** For pruning, quantization, and performance tuning of models.

**Concepts related to Keras:**

#### 1. Developing Sequential Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```

#### 2. Training and Validation
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

#### 3. Parameter Optimization
KerasTuner or grid search can be used for hyperparameter optimization (like learning rate, batch size, epochs).

---

### **6. Explain a simple Theano Program**
```python
import numpy
import theano.tensor as T
from theano import function

# Define two scalar variables
x = T.dscalar('x')
y = T.dscalar('y')

# Define operation
z = x + y

# Create a callable function
f = function([x, y], z)

# Execute
print(f(5, 7))  # Output: 12
```

**Explanation:**  
Theano uses symbolic variables and compiles the computation graph into efficient C code. Here, two scalars are added symbolically, and `function()` converts it into an executable Python function.

---

### **7. Explain PyTorch Tensors**

- **Definition:** Tensors are multi-dimensional arrays similar to NumPy arrays but with GPU acceleration.  
- They are the core data structure in PyTorch for building and training neural networks.

**Example:**
```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
print(x)
```

**Advantages:**
- Supports GPU operations for faster computation.
- Automatic differentiation through `torch.autograd`.

**Related Technologies:**
- **Uber’s Pyro:** A deep probabilistic programming framework built on PyTorch, used for Bayesian modeling and uncertainty estimation.
- **Tesla Autopilot:** Uses PyTorch for real-time perception, computer vision, and neural network inference in autonomous driving systems.

---

### **8. What is the use of NumPy, Pandas, and Sklearn in Deep Learning?**

- **NumPy:** For numerical operations and matrix manipulations.
- **Pandas:** For data manipulation and preprocessing using DataFrames.
- **Scikit-learn:** For data splitting, preprocessing, and classical ML models.

---

### **9. What are the main differences between TensorFlow and PyTorch?**

| Feature | TensorFlow | PyTorch |
|----------|-------------|---------|
| **Computation Graph** | Static graph | Dynamic graph |
| **Ease of Use** | Steeper learning curve | Pythonic and easier for beginners |
| **Deployment** | TensorFlow Lite, TensorFlow Serving | TorchScript, ONNX |
| **Community** | Larger, production-focused | Research-oriented and flexible |

---

### **10. What is the importance of Virtual Environments in Python?**
Virtual environments allow you to isolate dependencies for different projects, preventing version conflicts between libraries. Each environment has its own Python binaries and package versions.

---

### **11. Why do we upgrade pip before installation?**
Upgrading pip ensures compatibility with the latest package versions and prevents installation errors caused by outdated dependencies.

---

### **12. What is TensorBoard used for?**
TensorBoard is a visualization tool for TensorFlow. It helps monitor training metrics such as loss, accuracy, and model graph visualizations.

---

### **13. What is a Sequential model in Keras?**
Sequential models are simple linear stacks of layers where data flows sequentially from one layer to the next. It’s best suited for feed-forward networks.

---

### **14. What is GPU acceleration in PyTorch?**
PyTorch supports GPU acceleration through CUDA, allowing faster tensor computations by offloading them to GPU instead of CPU.

---

### **15. What is AutoKeras used for?**
AutoKeras automates the process of model selection and hyperparameter tuning using AutoML techniques.

---

### **16. What is TensorFlow Lite used for?**
TensorFlow Lite is a lightweight version of TensorFlow designed for running models on mobile and embedded devices.

---

### **17. What is the difference between TensorFlow and Keras?**
- TensorFlow is a full deep learning framework.
- Keras is a high-level API built on top of TensorFlow to simplify model development.

---

### **18. What is the role of Theano in Deep Learning history?**
Theano was one of the first deep learning frameworks that introduced symbolic computation and GPU support. It laid the foundation for frameworks like TensorFlow and Keras.

---

### **19. What is the use of PyTorch Autograd?**
Autograd in PyTorch automatically computes gradients, making backpropagation easier for training neural networks.

---

### **20. What is a pretrained model?**
A pretrained model is a model that has already been trained on a large dataset (like ImageNet) and can be fine-tuned for specific tasks, saving training time and improving accuracy.

---

**Conclusion:**  
All four deep learning frameworks — TensorFlow, Keras, Theano, and PyTorch — are essential tools for building neural networks. Their selection depends on the application domain, model complexity, and deployment requirements.

---
