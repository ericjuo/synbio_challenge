import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    T5Tokenizer, T5Model, BertModel, BertTokenizer,
    AutoModelForMaskedLM
)
from tokenizers import Tokenizer
import re
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture
class ProteinNet(nn.Module):
    def __init__(self, input_size):
        super(ProteinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def get_embeddings(sequences, model_name, batch_size=16):
    """Get embeddings for a list of sequences using a specific model."""
    print(f"\nProcessing model: {model_name}")
    
    # Filter out None and invalid sequences
    valid_sequences = []
    for seq in sequences:
        if isinstance(seq, str) and len(seq) > 0 and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in seq):
            valid_sequences.append(seq)
        else:
            print(f"Warning: Skipping invalid sequence: {seq}")
    
    if not valid_sequences:
        print("No valid sequences found!")
        return None
    
    print(f"Processing {len(valid_sequences)} valid sequences out of {len(sequences)} total sequences")
    
    # Special handling for different model types
    if model_name.startswith("hugohrban/progen2"):
        # Progen models
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = Tokenizer.from_pretrained(model_name)
            tokenizer.no_padding()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading Progen model {model_name}: {e}")
            return None

        embeddings = []
        num_sequences = len(valid_sequences)
        num_batches = (num_sequences + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, num_sequences, batch_size):
                batch_seqs = valid_sequences[i:i + batch_size]
                batch_embeddings = []
                
                try:
                    for seq in batch_seqs:
                        input_ids = torch.tensor(tokenizer.encode(seq).ids).to(device)
                        outputs = model(input_ids, output_hidden_states=True)
                        # Get the last hidden state and perform average pooling
                        last_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
                        seq_embedding = torch.mean(last_hidden_state, dim=0)  # Shape: [hidden_size]
                        batch_embeddings.append(seq_embedding)
                    
                    batch_embeddings = torch.stack(batch_embeddings)
                    embeddings.append(batch_embeddings.cpu())
                    
                except Exception as e:
                    print(f"Error processing Progen batch: {e}")
                    embed_dim = model.config.hidden_size
                    error_embeddings = torch.zeros((len(batch_seqs), embed_dim))
                    embeddings.append(error_embeddings)

    elif model_name.startswith("Rostlab/prot_t5"):
        # T5 models
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
            model = T5Model.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading T5 model {model_name}: {e}")
            return None

        embeddings = []
        num_sequences = len(valid_sequences)
        num_batches = (num_sequences + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, num_sequences, batch_size):
                batch_seqs = valid_sequences[i:i + batch_size]
                try:
                    # Process sequences - add spaces between amino acids and replace rare/ambiguous AAs
                    batch_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch_seqs]
                    
                    # Tokenize sequences and pad to longest in batch
                    ids = tokenizer.batch_encode_plus(batch_seqs, add_special_tokens=True, padding="longest")
                    input_ids = torch.tensor(ids['input_ids']).to(device)
                    attention_mask = torch.tensor(ids['attention_mask']).to(device)
                    #decoder_input_ids = torch.tensor(ids['decoder_input_ids']).to(device)
                    decoder_input_ids = input_ids
                    # Get embeddings
                    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
                    #print(embedding_repr)
                    # Extract embeddings for each sequence in batch, removing padding and special tokens
                    batch_embeddings = []
                    for j, seq in enumerate(batch_seqs):
                        # Get sequence length (excluding padding and special tokens)
                        seq_len = len(seq.split())
                        # Extract embeddings for actual sequence length
                        seq_embedding = embedding_repr.last_hidden_state[j, :seq_len]
                        # Calculate mean embedding for the sequence
                        seq_embedding = seq_embedding.mean(dim=0)
                        batch_embeddings.append(seq_embedding)
                    
                    batch_embeddings = torch.stack(batch_embeddings)
                    embeddings.append(batch_embeddings.cpu())
                    
                except Exception as e:
                    print(f"Error processing T5 batch: {e}")
                    embed_dim = model.config.hidden_size
                    error_embeddings = torch.zeros((len(batch_seqs), embed_dim))
                    embeddings.append(error_embeddings)

    elif model_name.startswith("Rostlab/prot_bert"):
        # BERT models
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
            model = BertModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading BERT model {model_name}: {e}")
            return None

        embeddings = []
        num_sequences = len(valid_sequences)
        num_batches = (num_sequences + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, num_sequences, batch_size):
                batch_seqs = valid_sequences[i:i + batch_size]
                try:
                    # Process sequences
                    batch_seqs = [re.sub(r"[UZOB]", "X", seq) for seq in batch_seqs]
                    encoded_input = tokenizer(batch_seqs, padding=True, return_tensors='pt')
                    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                    
                    # Get embeddings
                    output = model(**encoded_input)
                    batch_embeddings = output.last_hidden_state.mean(dim=1).cpu()
                    embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    print(f"Error processing BERT batch: {e}")
                    embed_dim = model.config.hidden_size
                    error_embeddings = torch.zeros((len(batch_seqs), embed_dim))
                    embeddings.append(error_embeddings)

    elif model_name.startswith("proteinglm"):
        # ProteinGLM models
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
            model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading ProteinGLM model {model_name}: {e}")
            return None

        embeddings = []
        num_sequences = len(valid_sequences)

        with torch.inference_mode():
            for seq in valid_sequences:
                try:
                    # Process single sequence
                    output = tokenizer(seq, add_special_tokens=True, return_tensors='pt')
                    inputs = {
                        "input_ids": output["input_ids"].to(device),
                        "attention_mask": output["attention_mask"].to(device)
                    }
                    
                    # Get embeddings
                    output_embeddings = model(**inputs, output_hidden_states=True, return_last_hidden_state=True)
                    # Get all hidden states except the last one (to exclude <eos> token) and take first token
                    seq_embedding = output_embeddings.hidden_states[:-1, 0]
                    # Cast to float32 and move to CPU
                    seq_embedding = seq_embedding.to(torch.float32).cpu()
                    # Take mean across sequence length
                    seq_embedding = seq_embedding.mean(dim=0)
                    embeddings.append(seq_embedding)
                    
                except Exception as e:
                    print(f"Error processing sequence: {e}")
                    embed_dim = model.config.hidden_size
                    error_embedding = torch.zeros(embed_dim, dtype=torch.float32)
                    embeddings.append(error_embedding)

        # Stack all embeddings
        if embeddings:
            embeddings = torch.stack(embeddings)

    else:
        # Default handling for other models
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None

        embeddings = []
        num_sequences = len(valid_sequences)
        num_batches = (num_sequences + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, num_sequences, batch_size):
                batch_seqs = valid_sequences[i:i + batch_size]
                try:
                    inputs = tokenizer(batch_seqs, padding=True, truncation=True, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    else:
                        batch_embeddings = outputs[0].mean(dim=1)
                    
                    embeddings.append(batch_embeddings.cpu())
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    embed_dim = model.config.hidden_size
                    error_embeddings = torch.zeros((len(batch_seqs), embed_dim))
                    embeddings.append(error_embeddings)

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # if not embeddings:
    #     return None
        
    # Concatenate all batch embeddings
    try:
        if model_name.startswith("proteinglm"):
            full_embeddings = embeddings
            return full_embeddings.numpy()
        else:
            full_embeddings = torch.cat(embeddings, dim=0)
            return full_embeddings.numpy()
    except Exception as e:
        print(f"Error concatenating embeddings: {e}")
        return None
    
def get_one_hot_encoding(sequences):
    """Generate one-hot encoding for protein sequences."""
    print(sequences)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    
    # Find the maximum sequence length
    max_len = max(len(seq) for seq in sequences)
    print(max_len)
    
    # Initialize the one-hot encoding array
    one_hot = np.zeros((len(sequences), max_len, len(amino_acids)))
    
    # Fill the one-hot encoding array
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if aa in aa_to_idx:
                one_hot[i, j, aa_to_idx[aa]] = 1
    
    # Flatten the one-hot encoding to 2D array
    return one_hot.reshape(len(sequences), -1)

def read_fasta(file_path):
    """Read sequences from a FASTA file."""
    sequences = {}
    current_name = None
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    
    if current_name:
        sequences[current_name] = ''.join(current_seq)
    
    return sequences


def predict_brightness(model, X, device):
    """Make predictions using the PyTorch model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy()

def main():
    # Define models to use
    MODELS = [
        # "facebook/esm1v_t33_650M_UR90S_1",
        # "facebook/esm1v_t33_650M_UR90S_2",
        # "facebook/esm1v_t33_650M_UR90S_3",
        # "facebook/esm1v_t33_650M_UR90S_4",
        # "facebook/esm1v_t33_650M_UR90S_5",
        "facebook/esm2_t6_8M_UR50D",
        "hugohrban/progen2-large",
        # "hugohrban/progen2-BFD90",
        # "proteinglm/proteinglm-1b-mlm"
    ]
    
    # Read sequences from FASTA file
    sequences = read_fasta('./Inference_data/GFP_Round8_sb.fasta')
    print(sequences)
    # Prepare data for prediction
    names = list(sequences.keys())
    seqs = list(sequences.values())
    
    # Get embeddings for each model
    X = None
    for model_name in MODELS:
        print(f"\nProcessing model: {model_name}")
        X_single = get_embeddings(seqs, model_name)
        if X is None:
            X = X_single
        else:
            X = np.concatenate((X, X_single), axis=1)

    # Add one-hot encoding features
    X_one_hot = get_one_hot_encoding(seqs)
    print(f"One-hot encoding shape: {X_one_hot.shape}")
    X = np.concatenate((X, X_one_hot), axis=1)
    print(f"Final feature shape: {X.shape}")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model = ProteinNet(X.shape[1]).to(device)
    model.load_state_dict(torch.load('./models/best_model_DL_3nn_nodrop_ep200.pth', map_location=device))
    # Make predictions
    predictions = predict_brightness(model, X, device)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Name': names,
        'Sequence': seqs,
        'Predicted_Brightness': predictions.flatten()  # Flatten predictions to 1D array
    })
    
    # Save results to CSV
    # results_df.to_csv('Inference_data/predictions_GFP_Round3_v3_new_unresize.csv', index=False)
    results_df.to_csv('Inference_data/predictions_GFP_Round8_sb_2.csv', index=False)
    print("\nPredictions saved to Inference_data/GFP_Round8_sb_2.csv")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Number of sequences processed: {len(sequences)}")
    print("\nTop 5 brightest predictions:")
    print(results_df.nlargest(5, 'Predicted_Brightness')[['Name', 'Predicted_Brightness']])
    print("\nBottom 5 predictions:")
    print(results_df.nsmallest(5, 'Predicted_Brightness')[['Name', 'Predicted_Brightness']])

if __name__ == "__main__":
    main()
