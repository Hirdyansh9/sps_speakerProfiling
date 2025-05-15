import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# Audio Processing
SAMPLE_RATE = 16000
CLIP_SECONDS = 4.0
WAV_LEN = int(SAMPLE_RATE * CLIP_SECONDS)  # 64000 samples

# WavLM
PRETRAINED_SSL_MODEL = 'microsoft/wavlm-base-plus'
SSL_OUTPUT_DIM = 768

# Conformer Encoder Parameters
CONFORMER_INPUT_DIM = SSL_OUTPUT_DIM  # 768
CONFORMER_NUM_BLOCKS = 4
CONFORMER_FF_DIM_FACTOR = 4
CONFORMER_NHEAD = 8
CONFORMER_CONV_KERNEL_SIZE = 31
CONFORMER_DROPOUT = 0.1

# Attentive Statistics Pooling Parameters
ATTN_POOL_DIM = 128

# Normalization Stats (from training)
NORM_STATS = {
    'age': {'mean': 30.29, 'std': 7.77},
    'height': {'mean': 175.75, 'std': 9.52}
}

# Gender Mapping
GENDER_MAP = {'F': 0, 'M': 1}

# Task Configuration
TASKS = {
    'age': {'type': 'regression', 'loss_weight': 0.4},
    'Gender': {'type': 'classification', 'loss_weight': 0.3, 'num_classes': 2},
    'height': {'type': 'regression', 'loss_weight': 0.3}
}

# Head Configurations
HEAD_CONFIGS = {
    'age': {'head_hidden_dim': 128, 'head_dropout_rate': 0.25},
    'Gender': {'head_hidden_dim': 96, 'head_dropout_rate': 0.25},
    'height': {'head_hidden_dim': 64, 'head_dropout_rate': 0.2}
}


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation: nn.Module = Swish(), bias: bool = True):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=bias)
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, channels, time)
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)
        x_act, x_gate = x.chunk(2, dim=1)  # (batch, channels, time) each
        x = x_act * torch.sigmoid(x_gate) # GLU
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2) # (batch, time, channels)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: nn.Module = Swish()):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, conv_kernel_size: int, dropout: float):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # FFNN 1
        residual = x
        x = self.norm1(x)
        x = self.ffn1(x)
        x = self.dropout(x) * 0.5 + residual # Half-step residual

        # Multi-Head Self-Attention Part
        residual = x
        x = self.norm2(x)
        x_attn, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.dropout(x_attn) + residual

        # Convolution Part
        residual = x
        x = self.norm3(x)
        x = self.conv_module(x)
        x = self.dropout(x) + residual

        # FFNN 2
        residual = x
        x = self.norm4(x)
        x = self.ffn2(x)
        x = self.dropout(x) * 0.5 + residual # Half-step residual
        return x

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # x shape: (batch, seq_len, input_dim)
        # attention_mask shape: (batch, seq_len), 1 for valid, 0 for pad

        attn_weights = self.attention_mlp(x).squeeze(-1)  # (batch, seq_len)

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9) # Mask before softmax

        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len)
        attn_weights_expanded = attn_weights.unsqueeze(-1) # (batch, seq_len, 1)

        weighted_mean = torch.sum(x * attn_weights_expanded, dim=1) # (batch, input_dim)

        # Weighted standard deviation
        weighted_var = torch.sum((x**2) * attn_weights_expanded, dim=1) - weighted_mean**2
        weighted_std = torch.sqrt(weighted_var.clamp(min=1e-9)) # (batch, input_dim)

        # Concatenate mean and std
        pooled_output = torch.cat((weighted_mean, weighted_std), dim=1) # (batch, 2 * input_dim)
        return pooled_output

class PadCrop:
    def __init__(self, length, mode='eval'):
        self.length = length
        self.mode = mode

    def __call__(self, wav):
        current_len = wav.shape[-1]
        if current_len == self.length:
            return wav
        elif current_len > self.length:
            start = (current_len - self.length) // 2  # Center-crop for eval
            return wav[..., start:start + self.length]
        else:
            pad_width = self.length - current_len
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            return torch.nn.functional.pad(wav, (pad_left, pad_right), mode='constant', value=0.0)

class sps_ConformerWavLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PRETRAINED_SSL_MODEL, trust_remote_code=True)
        self.wavlm = WavLMModel.from_pretrained(PRETRAINED_SSL_MODEL, trust_remote_code=True)

        self.conformer_encoder = nn.Sequential(
            *[ConformerBlock(
                d_model=CONFORMER_INPUT_DIM,
                n_head=CONFORMER_NHEAD,
                d_ff=CONFORMER_INPUT_DIM * CONFORMER_FF_DIM_FACTOR,
                conv_kernel_size=CONFORMER_CONV_KERNEL_SIZE,
                dropout=CONFORMER_DROPOUT
              ) for _ in range(CONFORMER_NUM_BLOCKS)]
        )

        self.pooling = AttentiveStatisticsPooling(CONFORMER_INPUT_DIM, ATTN_POOL_DIM)
        pooled_output_dim = CONFORMER_INPUT_DIM * 2

        self.heads = nn.ModuleDict()
        for task_name, task_info in TASKS.items():
            head_config = HEAD_CONFIGS[task_name]
            output_dim = 1 if task_info['type'] == 'regression' else task_info.get('num_classes')

            self.heads[task_name] = nn.Sequential(
                nn.Linear(pooled_output_dim, head_config['head_hidden_dim']), nn.ReLU(),
                nn.Dropout(head_config['head_dropout_rate']),
                nn.Linear(head_config['head_hidden_dim'], output_dim)
            )

    def forward(self, waveform: torch.Tensor):
        current_device = waveform.device
        inputs = self.feature_extractor(
            [wav.cpu().numpy() for wav in waveform],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="longest"
        )
        input_values = inputs.input_values.to(current_device)

        attention_mask_ssl_input = inputs.attention_mask.to(current_device) if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None else None

        wavlm_outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask_ssl_input,
            output_hidden_states=False
        )
        hidden_states = wavlm_outputs.last_hidden_state

        # Handle padding mask for Conformer
        if attention_mask_ssl_input is not None:
            output_seq_len = hidden_states.shape[1]
            if attention_mask_ssl_input.shape[1] > output_seq_len:
                 conformer_padding_mask = (attention_mask_ssl_input[:, :output_seq_len] == 0)
            elif attention_mask_ssl_input.shape[1] < output_seq_len:
                 conformer_padding_mask = torch.zeros(hidden_states.shape[0], output_seq_len, dtype=torch.bool, device=current_device)
            else:
                 conformer_padding_mask = (attention_mask_ssl_input == 0)
        else:
            conformer_padding_mask = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bool, device=current_device)

        conformer_output = hidden_states
        for block in self.conformer_encoder:
            conformer_output = block(conformer_output, src_key_padding_mask=conformer_padding_mask)

        # For AttentiveStatisticsPooling, we need a mask where 1 is valid, 0 is pad
        pooling_attention_mask = ~conformer_padding_mask

        pooled_output = self.pooling(conformer_output, attention_mask=pooling_attention_mask)

        predictions = {}
        for task_name, head_module in self.heads.items():
            task_prediction = head_module(pooled_output)
            if TASKS[task_name]['type'] == 'regression':
                predictions[task_name] = task_prediction.squeeze(-1)
            else:
                predictions[task_name] = task_prediction
        return predictions

def load_model(model_path):
    # Initialize model
    model = sps_ConformerWavLM()
    
    # Load the saved weights (state_dict)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    
    # Set to evaluation mode
    model.eval()
    
    return model

def process_audio(audio_path, model, device='cpu'):
    # Load and preprocess audio
    wav, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # Pad or crop to expected length
    if wav.shape[1] > WAV_LEN:
        # Take center part
        start = (wav.shape[1] - WAV_LEN) // 2
        wav = wav[:, start:start+WAV_LEN]
    elif wav.shape[1] < WAV_LEN:
        # Pad
        pad_len = WAV_LEN - wav.shape[1]
        wav = F.pad(wav, (pad_len//2, pad_len - pad_len//2))
    
    # Move to device
    wav = wav.to(device)
    
    # Make prediction
    with torch.no_grad():
        predictions = model(wav)
    
    # Post-process predictions
    results = {}
    for task_name, pred in predictions.items():
        if TASKS[task_name]['type'] == 'regression':
            # Denormalize regression outputs
            mean, std = NORM_STATS[task_name]['mean'], NORM_STATS[task_name]['std']
            value = (pred.cpu().item() * std) + mean
            results[task_name] = value
        else:
            # For classification, get class label
            class_idx = torch.argmax(pred, dim=1).cpu().item()
            if task_name == 'Gender':
                gender_labels = {v: k for k, v in GENDER_MAP.items()}
                results[task_name] = gender_labels[class_idx]
            else:
                results[task_name] = class_idx
    
    return results

# Example usage
def predict(model_path, audio_path, device='cpu'):
    model = load_model(model_path)
    model = model.to(device)
    results = process_audio(audio_path, model, device)
    
    # Format and return results
    print(f"Age: {results['age']:.1f} years")
    print(f"Gender: {results['Gender']}")
    print(f"Height: {results['height']:.1f} cm")
    
    return results