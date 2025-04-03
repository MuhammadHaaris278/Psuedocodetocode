import streamlit as st
st.set_page_config(page_title="Pseudocode to C++ Converter", page_icon="🖥", layout="centered")
import torch
import torch.nn as nn
import sentencepiece as spm
import math
import torch.serialization
import torch.nn.functional as F
import time


# Load Tokenizers
sp_pseudo = spm.SentencePieceProcessor()
sp_code = spm.SentencePieceProcessor()
sp_pseudo.load("pseudo.model")  # Pseudocode tokenizer
sp_code.load("code.model")  # C++ code tokenizer

# Define Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Head dimension

        # ✅ Use exact names as in the saved model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_output))

# Feed-Forward Layer
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()  # ✅ Matches trained model (no dropout)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # ✅ Matches trained model


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # ✅ FIXED: Matches trained model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)  # ✅ FIXED: Matches trained model
        return self.norm2(x + self.dropout(ff_output))

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # ✅ FIXED: Matches trained model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)  # ✅ FIXED: Matches trained model
        return self.norm3(x + self.dropout(ff_output))

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, None, None)
        return self.fc(dec_output)

# Load Model
@st.cache_resource
def load_model():
    print("🔄 Attempting to load model...")  # ✅ Debugging
    torch.serialization.add_safe_globals([MultiHeadAttention])
    
    model = torch.load("transformer_full_new.pth", map_location=torch.device("cpu"), weights_only=False)  
    model.eval()
    
    print("✅ Model Loaded Successfully:", model)  # ✅ Confirm it prints
    return model


# Ensure model is stored in session state
if "model" not in st.session_state:
    st.session_state.model = load_model()

# Translation Function
def translate(pseudocode, beam_size=5, max_length=100):
    model = st.session_state.model
    model.eval()

    # Encode input pseudocode
    tokens = sp_pseudo.encode(pseudocode, out_type=int)
    print("🔹 Tokenized Input:", tokens)  # Debugging: Print tokenized input

    src_tensor = torch.tensor(tokens).unsqueeze(0)
    tgt_tensor = torch.tensor([[sp_code.bos_id()]], dtype=torch.long)

    sequences = [(tgt_tensor, 0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            output = model(src_tensor, seq)
            next_token_probs = F.log_softmax(output[:, -1, :], dim=-1)
            top_k_probs, top_k_tokens = torch.topk(next_token_probs, beam_size)

            for i in range(beam_size):
                next_token = top_k_tokens[:, i].unsqueeze(1)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + top_k_probs[:, i].item()

                if next_token.item() == sp_code.eos_id():
                    print("✅ Final Tokenized Output:", new_seq.squeeze().tolist())  # Debugging
                    print("✅ Decoded Output:", sp_code.decode(new_seq.squeeze().tolist()))  # Debugging
                    return sp_code.decode(new_seq.squeeze().tolist())

                all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    pred_tokens = sequences[0][0].squeeze().tolist()
    print("✅ Final Tokenized Output:", pred_tokens)  # Debugging
    print("✅ Decoded Output:", sp_code.decode(pred_tokens))  # Debugging
    return sp_code.decode(pred_tokens)

# Streamlit UI
# Custom CSS for better UI
st.markdown("""
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #E63946;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #A8A8A8;
            text-align: center;
        }
        .stTextArea textarea {
            background-color: #2E2E2E !important;
            color: white !important;
        }
        .stButton button {
            background: linear-gradient(to right, #E63946, #FF6B6B);
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(to right, #FF6B6B, #E63946);
            transform: scale(1.05);
        }
        .code-box {
            border: 2px solid #E63946;
            border-radius: 10px;
            padding: 10px;
            background: #2E2E2E;
            color: white;
            font-family: 'Courier New', monospace;
            font-size: 16px;
        }
        .copy-btn {
            background-color: #E63946;
            color: white;
            font-size: 14px;
            font-weight: bold;
            border-radius: 5px;
            padding: 5px 10px;
            margin-top: 5px;
            cursor: pointer;
            transition: 0.2s;
        }
        .copy-btn:hover {
            background-color: #FF6B6B;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 class='title'>Pseudocode to C++ Translator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Convert structured pseudocode into optimized C++ code using AI.</p>", unsafe_allow_html=True)

# User Input
pseudocode = st.text_area("Enter Your Pseudocode Below 👇", height=200)

# Character Counter
char_count = len(pseudocode)
st.write(f"📝 **Character Count:** {char_count} / 1000")

# Translate Button
if st.button("🔄 Translate"):
    if pseudocode.strip():
        with st.spinner("🚀 Generating C++ Code... Please wait..."):
            time.sleep(2)  # Simulating processing delay
            cpp_code = translate(pseudocode)
        
        # Display Result
        st.subheader("✅ Generated C++ Code:")
        st.markdown(f"<div class='code-box'><pre>{cpp_code}</pre></div>", unsafe_allow_html=True)

        # Copy Button
        st.markdown(
            f"<button class='copy-btn' onclick='navigator.clipboard.writeText(`{cpp_code}`)'>📋 Copy Code</button>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter valid pseudocode before translating.")