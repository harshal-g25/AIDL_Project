import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepDTA(nn.Module):
    def __init__(self, drug_fingerprint_size, prot_max_len, vocab_size, embedding_dim):
        super(DeepDTA, self).__init__()

        # Drug Encoder (Dense Layers)
        self.drug_net = nn.Sequential(
            nn.Linear(drug_fingerprint_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Protein Encoder (1D CNN)
        self.protein_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.prot_net = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Regression Head (Feature Fusion + Fully Connected Layers)
        conv_output_size = self._get_conv_output_size(prot_max_len, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(512 + conv_output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def _get_conv_output_size(self, prot_max_len, embedding_dim):
        x = torch.randn(1, prot_max_len, embedding_dim).permute(0, 2, 1)
        x = self.prot_net(x)
        return x.size(0) * x.size(1) * x.size(2)

    def forward(self, drug_input, protein_input):
        drug_features = self.drug_net(drug_input)

        protein_embedded = self.protein_embedding(protein_input).permute(0, 2, 1)
        protein_features = self.prot_net(protein_embedded)
        protein_features = protein_features.view(protein_features.size(0), -1)

        combined_features = torch.cat((drug_features, protein_features), dim=1)
        affinity_prediction = self.head(combined_features)
        
        return affinity_prediction

# Define your model parameters (adjust as per your dataset)
drug_fingerprint_size = 2048  # Example
prot_max_len = 1000          # Example
vocab_size = 26              # Example (20 amino acids + padding tokens)
embedding_dim = 128          # Example