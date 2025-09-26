# NSCLC Radiogenomics Classification

Multimodal lung cancer classification using CT and PET imaging data.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run quick test:**
   ```bash
   python quick_test.py
   ```

3. **Generate code with AI:**
   - Copy templates from the guide into Cursor/VSCode
   - Use AI prompts to complete the code
   - Test locally before deploying to Colab

## Project Structure

```
├── config/                 # Configuration files
├── src/                   # Source code
│   ├── data/             # Data loading and processing
│   ├── models/           # Model architectures
│   ├── training/         # Training utilities
│   └── utils/           # Helper utilities
├── test_components/      # Local testing scripts
├── notebooks/           # Jupyter notebooks for Colab
├── logs/               # Training logs
└── requirements.txt    # Dependencies
```

## Testing Workflow

1. **Local Testing:**
   ```bash
   # Test individual components
   python test_components/test_metadata_handler.py
   python test_components/test_dicom_processor.py
   
   # Run full test suite
   python run_local_tests.py
   ```

2. **Colab Deployment:**
   - Upload code to GitHub
   - Clone in Colab
   - Mount Google Drive
   - Run training script

## AI Development Tips

- Use specific prompts with error messages and data shapes
- Test AI-generated code on small samples first
- Iterate quickly with AI for optimization and debugging
- Let AI handle boilerplate while you focus on architecture
