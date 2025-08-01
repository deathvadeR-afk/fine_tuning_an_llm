# Fine-Tuning TinyLlama on SMS Spam Classification

This project demonstrates how to fine-tune the TinyLlama-1.1B-Chat-v1.0 model for SMS spam classification using Parameter Efficient Fine-Tuning (PEFT) techniques, specifically LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA).

## Overview

The project uses the SMS Spam Collection dataset to train a lightweight language model to classify text messages as either "spam" or "ham" (legitimate messages). By leveraging modern fine-tuning techniques, we achieve excellent performance while keeping computational requirements minimal.

## Key Features

- **Efficient Fine-Tuning**: Uses LoRA adapters to train only ~0.2% of model parameters
- **Memory Optimization**: Implements 4-bit quantization with QLoRA to reduce memory usage
- **High Performance**: Achieves 100% accuracy on the test set
- **Practical Implementation**: Ready-to-use code for SMS spam detection

## Technical Approach

### Model Architecture
- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Task**: Sequence Classification (Binary)
- **Fine-tuning Method**: LoRA with 4-bit quantization

### Dataset
- **Source**: SMS Spam Collection dataset
- **Classes**: Binary classification (ham/spam)
- **Split**: 80% training, 20% testing
- **Features**: Text messages with corresponding labels

### Training Configuration
- **LoRA Parameters**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, v_proj
  - Dropout: 0.05
- **Training Parameters**:
  - Batch size: 16
  - Learning rate: 2e-4
  - Epochs: 3
  - Gradient accumulation steps: 2

## Requirements

```bash
pip install transformers datasets peft bitsandbytes accelerate torch
```

### Hardware Requirements
- GPU with CUDA support (recommended)
- Minimum 8GB GPU memory (with quantization)
- Google Colab T4 GPU works well

## Project Structure

```
├── Fine_tuning_tinyllama_on_smsSPAM.ipynb  # Main notebook
├── README.md                               # This file
└── tinyllama-sms-spam/                     # Output directory (created during training)
```

## Usage

### Running the Notebook
1. Open the Jupyter notebook in Google Colab or your local environment
2. Install required dependencies
3. Run cells sequentially to:
   - Load and prepare the dataset
   - Configure the model with quantization
   - Apply LoRA adapters
   - Train the model
   - Evaluate performance

### Key Steps Explained

1. **Environment Setup**: Install necessary libraries and authenticate with Hugging Face
2. **Model Loading**: Load TinyLlama with 4-bit quantization for memory efficiency
3. **Dataset Preparation**: Load SMS spam dataset and tokenize for model input
4. **LoRA Configuration**: Set up parameter-efficient fine-tuning adapters
5. **Training**: Fine-tune only the adapter parameters
6. **Evaluation**: Test model performance and run inference examples

### Example Usage

```python
# Test with new SMS messages
def predict_spam(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    return "Spam" if torch.argmax(logits) == 1 else "Not Spam"

# Example predictions
predict_spam("WINNER!! You've been selected for a free iPhone!!!")  # Returns: "Spam"
predict_spam("Hey, are we still meeting for lunch today?")           # Returns: "Not Spam"
```

## Results

- **Test Accuracy**: 100%
- **Trainable Parameters**: 2,256,896 out of 1,036,773,376 (0.22%)
- **Training Time**: Approximately 10-15 minutes on T4 GPU
- **Memory Usage**: Significantly reduced due to 4-bit quantization

## Technical Benefits

### Parameter Efficiency
- Only trains adapter layers, not the entire model
- Reduces overfitting risk
- Faster training and inference

### Memory Optimization
- 4-bit quantization reduces model size by ~75%
- Enables training on consumer GPUs
- Maintains model performance

### Practical Applications
- Real-time SMS filtering
- Email spam detection
- Content moderation systems
- Educational demonstrations of modern NLP techniques

## Learning Outcomes

This project demonstrates:
- Modern parameter-efficient fine-tuning techniques
- Practical application of quantization for resource-constrained environments
- End-to-end workflow from data loading to model deployment
- Best practices for text classification with transformer models

## Future Enhancements

- Experiment with different LoRA configurations
- Test on larger datasets
- Implement multi-class classification
- Add model deployment pipeline
- Create web interface for real-time predictions

## References

- [TinyLlama Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [SMS Spam Collection Dataset](https://huggingface.co/datasets/sms_spam)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## License

This project is for educational purposes. Please check the licenses of the underlying models and datasets for commercial use.
