
# ===========================================
# CNN Model
# ===========================================

encoder = nn.Sequential(
    ConvBlock(9, 64, kernel_size=7),      
    ConvBlock(64, 96, kernel_size=5),    
    nn.Conv1d(96, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
)

classifier = nn.Linear(128, 6, bias=False)

h = encoder(x)                         
logits = classifier(h.mean(dim=-1))    


# ===========================================
# BiLSTM Model
# ===========================================

input_proj = nn.Linear(9, 64)

encoder = nn.LSTM(
    input_size=64,
    hidden_size=64,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.15,
)

norm = nn.LayerNorm(128)
classifier = nn.Linear(128, 6, bias=False)

z = input_proj(x.transpose(1, 2))       
z, _ = encoder(z)                     
h = norm(z).transpose(1, 2)          
logits = classifier(h.mean(dim=-1))     

# ===========================================
# Transformer
# ===========================================

input_proj = nn.Linear(9, 128)
positional_embedding = nn.Parameter(torch.zeros(1, 128, 128))

encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,
    dim_feedforward=256,
    dropout=0.10,
    activation="gelu",
    batch_first=True,
    norm_first=True,
)

encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
norm = nn.LayerNorm(128)
classifier = nn.Linear(128, 6, bias=False)

nn.init.normal_(positional_embedding, mean=0.0, std=0.02)

z = input_proj(x.transpose(1, 2))       
z = z + positional_embedding
h = norm(encoder(z)).transpose(1, 2)     
logits = classifier(h.mean(dim=-1))    





