# Use multiple embedding and combine them together to train a single model
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    T5Tokenizer, T5Model, BertModel, BertTokenizer,
    AutoModelForMaskedLM
)
from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

MODELS = [
    # ["facebook/esm1v_t33_650M_UR90S_1","facebook/esm1v_t33_650M_UR90S_2", "facebook/esm1v_t33_650M_UR90S_3", "facebook/esm1v_t33_650M_UR90S_4", "facebook/esm1v_t33_650M_UR90S_5","facebook/esm2_t6_8M_UR50D" ,"hugohrban/progen2-large", "hugohrban/progen2-BFD90", "proteinglm/proteinglm-1b-mlm"],
["facebook/esm2_t6_8M_UR50D" ,"hugohrban/progen2-large"]
]

class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ProteinNet(nn.Module):
    def __init__(self, input_size):
        super(ProteinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, 1)

        )
        
    def forward(self, x):
        return self.network(x)
    

def normalize_brightness(brightness_values):
    """Normalize brightness values to range [0,1]."""
    min_val = np.min(brightness_values)
    max_val = np.max(brightness_values)
    print(f"Original brightness range: [{min_val}, {max_val}]")
    normalized = (brightness_values - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


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
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    
    # Find the maximum sequence length
    max_len = max(len(seq) for seq in sequences)
    
    # Initialize the one-hot encoding array
    one_hot = np.zeros((len(sequences), max_len, len(amino_acids)))
    
    # Fill the one-hot encoding array
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if aa in aa_to_idx:
                one_hot[i, j, aa_to_idx[aa]] = 1
    
    # Flatten the one-hot encoding to 2D array
    return one_hot.reshape(len(sequences), -1)

def plot_regression(y_true, y_pred, title, save_path):
    """Create a regression plot of predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Calculate R² score
    r2 = r2_score(y_true, y_pred)
    
    # Add labels and title
    plt.xlabel('Actual Brightness')
    plt.ylabel('Predicted Brightness')
    plt.title(f'{title}\nR² = {r2:.4f}')
    plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate(X, y, model_name):
    """Train neural network and evaluate performance."""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinNet(X.shape[1]).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_r2 = -float('inf')
    best_model_state = None
    num_epochs = 200
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(batch_y.numpy())
        
        val_r2 = r2_score(val_true, val_preds)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val R²: {val_r2:.4f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_preds = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
        val_preds = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
    
    r2_train = r2_score(y_train, train_preds)
    r2_val = r2_score(y_val, val_preds)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the best model
    model_filename = f"models/best_model_DL_3nn_nodrop_ep200.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"\nBest model saved to {model_filename}")
    
    return r2_train, r2_val, model, y_train, train_preds, y_val, val_preds

def plot_results(results):
    """Plot R² scores for different models."""
    plt.figure(figsize=(15, 8))
    
    # Create DataFrame for plotting
    df = pd.DataFrame(results)
    
    # Melt DataFrame for seaborn
    df_melted = pd.melt(df, id_vars=['Model'], value_vars=['R² Train', 'R² Val'],
                       var_name='Metric', value_name='Score')
    
    # Create bar plot
    sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
    
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    #plt.savefig('./Train_result/PNG/Model_train_DL_3nn_nodrop_ep200_unsample.png')
    plt.close()

def main():
# Load your data
# Assuming you have a DataFrame with 'full_sequence' and 'Brightness' columns
    df = pd.read_excel('./GFP_data.xlsx', sheet_name='brightness')

    # Split data by GFP type
    avGFP_df = df[df['GFP type'] == 'avGFP'].copy()
    amacGFP_df = df[df['GFP type'] == 'amacGFP'].copy()
    cgreGFP_df = df[df['GFP type'] == 'cgreGFP'].copy()
    ppluGFP2_df = df[df['GFP type'] == 'ppluGFP'].copy()

    # Load WT sequences
    with open('./AAseqs of 4 GFP proteins.txt', 'r') as f:
        wt_seqs = {}
        current_type = None
        for line in f:
            if line.startswith('>'):
                current_type = line.strip()[1:]  # Remove '>' and store type
            elif current_type and line.strip():
                wt_seqs[current_type] = line.strip()

    # Generate full sequences if not already present
    def generate_mutated_sequence(mutation_str, wt_sequence):
        if mutation_str.strip().upper() == 'WT':
            return wt_sequence
            
        sequence = list(wt_sequence)
        mutations = mutation_str.split(':')
        
        for mut in mutations:
            match = re.match(r'([A-Z])(\d+)([A-Z*.])$', mut.strip(), re.IGNORECASE)
            if match:
                original_aa, pos, new_aa = match.groups()
                pos = int(pos) - 1
                
                if pos < 0 or pos >= len(sequence):
                    continue
                    
                if new_aa == '*':
                    return None
                elif new_aa == '.':
                    new_aa = sequence[pos]
                    
                sequence[pos] = new_aa.upper()
                
        return ''.join(sequence)

    # Generate sequences for each GFP type
    for gfp_type, gfp_df in [('avGFP', avGFP_df), ('amacGFP', amacGFP_df), 
                            ('cgreGFP', cgreGFP_df), ('ppluGFP2', ppluGFP2_df)]:
        gfp_df['full_sequence'] = gfp_df['aaMutations'].apply(
            lambda x: generate_mutated_sequence(x, wt_seqs[gfp_type])
        )
        gfp_df.dropna(subset=['full_sequence', 'Brightness'], inplace=True)
        gfp_df['Brightness'] = pd.to_numeric(gfp_df['Brightness'], errors='coerce')
        gfp_df.dropna(subset=['Brightness'], inplace=True)

    sampled_subsets = []
    sampled_subsets.append(avGFP_df)
    sampled_subsets.append(amacGFP_df)
    sampled_subsets.append(cgreGFP_df)
    sampled_subsets.append(ppluGFP2_df)


    #### This part is for the specific sampling method, taking the 10% brightest ####
    #### and the 10% least bright as the training dataset. This sampilng method  ####
    #### can also acheive testing R2 = 0.90 with faster training speed           ####
    
    # Define sampling function
    # def get_subsets(df):
    #     # Sort by brightness
    #     sorted_df = df.sort_values('Brightness')
        
    #     # Calculate 10% threshold
    #     n_samples = len(sorted_df)
    #     n_10_percent = int(n_samples * 0.1)
        
    #     # Get brightest and least bright 10%
    #     brightest = sorted_df.tail(n_10_percent)
    #     least_bright = sorted_df.head(n_10_percent)
        
    #     return brightest, least_bright

    # # Get subsets for each GFP variant
    # avGFP_brightest, avGFP_least = get_subsets(avGFP_df)
    # amacGFP_brightest, amacGFP_least = get_subsets(amacGFP_df)
    # cgreGFP_brightest, cgreGFP_least = get_subsets(cgreGFP_df)
    # ppluGFP2_brightest, ppluGFP2_least = get_subsets(ppluGFP2_df)

    # # List of all subsets
    # subsets = [
    #     avGFP_brightest, avGFP_least,
    #     amacGFP_brightest, amacGFP_least,
    #     cgreGFP_brightest, cgreGFP_least,
    #     ppluGFP2_brightest, ppluGFP2_least
    # ]
    # Sample 5000 sequences from each subset
    # sampled_subsets = []
    # for subset in subsets:
    #     if len(subset) >= 5000:
    #         sampled = subset.sample(n=5000, random_state=42)
    #     else:
    #         # If subset has fewer than 5000 sequences, use all of them
    #         sampled = subset
    #     sampled_subsets.append(sampled)

    # Combine all sampled subsets into final training dataframe
    sampled_train_df = pd.concat(sampled_subsets, ignore_index=True)
    
    # Print information about the final dataset
    print(f"Final sampled training set size: {len(sampled_train_df)}")

    #### Uncomment the following if you use sampling method  ####
    # print("\nDistribution of sequences:")
    # print(f"avGFP brightest: {len(sampled_subsets[0])}")
    # print(f"avGFP least bright: {len(sampled_subsets[1])}")
    # print(f"amacGFP brightest: {len(sampled_subsets[2])}")
    # print(f"amacGFP least bright: {len(sampled_subsets[3])}")
    # print(f"cgreGFP brightest: {len(sampled_subsets[4])}")
    # print(f"cgreGFP least bright: {len(sampled_subsets[5])}")
    # print(f"ppluGFP2 brightest: {len(sampled_subsets[6])}")
    # print(f"ppluGFP2 least bright: {len(sampled_subsets[7])}")

    # Prepare sequences and labels for embedding
    sequences = sampled_train_df['full_sequence'].tolist()
    y = sampled_train_df['Brightness'].values
    
    # Normalize brightness values
    y_normalized, min_brightness, max_brightness = normalize_brightness(y)
    print(f"Normalized brightness range: [{np.min(y_normalized)}, {np.max(y_normalized)}]")
    # Store results
    results = []
    best_models = {}
    
    # Process each model
    for model_name in MODELS:
        print(f"\nProcessing model: {model_name}")
        start_time = time.time()
        X = None
        # Get embeddings
        for single_model in model_name:
            X_single = get_embeddings(sequences, single_model)
            if X is None:
                X = X_single
            else:
                X = np.concatenate((X, X_single), axis=1)
        
        # Add one-hot encoding features
        X_one_hot = get_one_hot_encoding(sequences)
        X = np.concatenate((X, X_one_hot), axis=1)
        
        if X is not None:
            # Train and evaluate
            r2_train, r2_val, best_model, y_train, train_preds, y_val, val_preds = train_and_evaluate(X, y_normalized, model_name)
            
            # Create regression plots
            os.makedirs('./Train_result/PNG', exist_ok=True)

            # Denormalize predictions for plotting
            train_preds_denorm = train_preds * (max_brightness - min_brightness) + min_brightness
            val_preds_denorm = val_preds * (max_brightness - min_brightness) + min_brightness
            y_train_denorm = y_train * (max_brightness - min_brightness) + min_brightness
            y_val_denorm = y_val * (max_brightness - min_brightness) + min_brightness

            # Create regression plots with denormalized values
            # plot_regression(y_train_denorm, train_preds_denorm, 
            #             'Training Set: Predicted vs Actual Brightness',
            #             './Train_result/PNG/train_regression_DL_3nn_nodrop_ep200_unsample.png')
            # plot_regression(y_val_denorm, val_preds_denorm,
            #             'Validation Set: Predicted vs Actual Brightness',
            #             './Train_result/PNG/val_regression_DL_3nn_nodrop_ep200_unsample.png')
            # Store results
            results.append({
                'Model': model_name,
                'R² Train': r2_train,
                'R² Val': r2_val,
                'Time': time.time() - start_time
            })
            
            # Store best model
            best_models['_'.join(model_name)] = best_model
            
            print(f"R² Train: {r2_train:.4f}")
            print(f"R² Val: {r2_val:.4f}")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
        else:
            print(f"Failed to get embeddings for {model_name}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    # results_df.to_csv('./Train_result/CSV/Model_train_DL_3nn_nodrop_ep200.csv', index=False)
    
    # Plot results
    plot_results(results_df)
    
    return best_models

if __name__ == "__main__":
    best_models = main()