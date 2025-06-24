# 2025 SynBio Challenge 

## Overview
In Brightness_model folder, we demonstrate the complete model training flow that predicts the brightness of wildtype GFPs (avGFP, amacGFP, cgreGFP, ppluGFP) and their mutants using simple fully connected neural network model. Our method combines (1) Multiple protein language model embeddings (e.g., ESM2, ProGen2) and (2) One-hot encoding as input features to a neural network regressor. The model can be trained on whole GFP dataset or partial dataset using defined sampling strategy, depending on the user computing resource. After training, the model can be used to predict the brightness of new protein sequences.


## Method

### Feature Extraction  
Protein sequences are embedded using pre-trained protein language models (ESM2, ProGen2, etc.) from HuggingFace Transformers. Furthermore, One-hot encoding of sequences is also included for improving model performance.   

### Model Architecture  
A feedforward neural network (3 layers, ReLU activations) is trained to regress normalized brightness values.  

### Training Data  
The model is trained on given dataset of GFP variants and datasets can be sampled or split by GFP type. 

### Inference   
The trained model can predict the brightness of sequences provided in FASTA format. You can check the sample fasta format in Inference_data folder.  


## File Structure  
model_train_DL.py — Training script  
model_inference_DL.py — Inference script  
requirements.txt — Python dependencies  
models/ — Saved model weights  
Inference_data/ — Input FASTA files and prediction outputs  
Train_result/ — Training logs and plots  


##  Installation guide
1. Clone the repository
```
git clone <your_repo_url>
cd Brightness_model
```
2. Set up the environment  
```bash
conda create -n gfp_brightness python=3.8
conda activate gfp_brightness
pip install -r requirements.txt
```

## Usage  

1. Training  
Place the training data Excel file (e.g., GFP_data.xlsx) and wild-type sequence file (AAseqs of 4 GFP proteins.txt) under this directory.   
Adjust the file paths in model_train_DL.py if needed.   
Run training:   
``` bash
python model_train_DL.py
```

2. Inference
Place your FASTA file with protein sequences (e.g., GFP_Round8_sb.fasta) in the Inference_data directory. 
Run inference:     
``` bash
python model_inference_DL.py
```   
The script will load the trained model in models/ and output predictions to Inference_data/predictions_GFP_Round8_sb_2.csv.


## Notes
The first time you run the scripts, large pre-trained models will be downloaded from HuggingFace.  
GPU is recommended for faster embedding (espectially for progen2-large) and training.   
You can modify the list of protein pretrained models used for embeddings in the scripts (MODELS variable).  
