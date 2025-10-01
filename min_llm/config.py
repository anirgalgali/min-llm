from dataclasses import dataclass, field
from typing import Optional, Literal

PositionEmbeddingType = Literal["rope","absolute","none"]
NormPositionType = Literal["pre", "post"]
NormType = Literal["rms", "layer"]
ActivationType = Literal["gelu", "relu","silu"]

@dataclass
class SelfAttentionConfig:
    n_heads: int = 4
    bias: bool = False
    dropout_attn: float = 0.0

@dataclass
class FFNConfig:
    is_gated: bool = True
    activation: ActivationType = "silu"
    bias: bool = False
    hidden_dim_multiplier: Optional[float] = None
    # when none (set to defaults below):
    #   hidden_dim_multiplier = 8/3 (is_gated = True)
    #   hidden_dim_multiplier = 4 (is_gated = False)


@dataclass
class TransformerConfig:
    norm_position: NormPositionType = "pre" # whether to have normalizaiton before or after computations
    norm_type: NormType = "layer"  # rms or layernorm
    use_causal_mask: bool = True
    ln_bias: bool = False  # bias for layer norm layers
    ffn: FFNConfig = field(default_factory=FFNConfig)
    attn: SelfAttentionConfig = field(default_factory=SelfAttentionConfig)


@dataclass
class ArchitectureConfig:
    d_model: int
    vocab_size: int
    context_length: int

@dataclass
class DistilBERTConfig(ArchitectureConfig):
    d_model: int = 768
    n_layers: int = 6
    vocab_size: int = 30522
    context_length: int = 512
    pos_embedding_type: PositionEmbeddingType = "absolute"
    dropout: float = 0.1
    transformer: Optional[TransformerConfig] = None 

    def __post_init__(self):
        if self.transformer is None:

            self.transformer = TransformerConfig(norm_position="post",
                                                 norm_type = "layer",
                                                 use_causal_mask=False,
                                                 ln_bias = True,
                                                 ffn = FFNConfig(
                                                 is_gated=False,
                                                 activation = "gelu",
                                                 bias = True),
                                                 attn = SelfAttentionConfig(
                                                 n_heads=12,
                                                 bias = True))

@dataclass
class DecoderLMConfigTest(ArchitectureConfig):
    n_layers: int
    theta: float
    transformer: TransformerConfig
    pos_embedding_type: PositionEmbeddingType = "rope"
    dropout: float = 0.0
    share_embed_lmhead_wts: bool = False


@dataclass
class DecoderLMConfig(ArchitectureConfig):
    d_model: int = 512
    theta: float = 10000.0
    n_layers: int = 4
    vocab_size: int = 50257
    context_length: int = 256
    pos_embedding_type: PositionEmbeddingType = "rope"
    share_embed_lmhead_wts: bool = True
    dropout: float = 0.1
    transformer: Optional[TransformerConfig] = None 

    def __post_init__(self):
        if self.transformer is None:

            self.transformer = TransformerConfig(norm_position="pre",
                                                 norm_type = "rms",
                                                 use_causal_mask=True,
                                                 ln_bias = False,
                                                 ffn = FFNConfig(
                                                 is_gated=True,
                                                 activation = "silu",
                                                 bias = False),
                                                 attn = SelfAttentionConfig(
                                                 n_heads=16,
                                                 bias = False))
        

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = None
    beta1: float = 0.90,   # taken from LLama
    beta2: float = 0.95,   # taken from LLama
    num_iterations: int = 2500
    patience_threshold: int = 3
    batch_size: int = 512
    resume_from_checkpoint: str = None
    reset_scheduler_on_load: bool = False
    eval_every_n_steps: int = 100
    min_lr: float = 0.0001
    num_warmup_steps: int = 250
    num_cosine_steps: int = 2000
    max_grad_norm: float = 1.0
        
@dataclass
class RunConfig:
    model:DecoderLMConfig = field(default_factory=DecoderLMConfig)
    train:TrainingConfig = field(default_factory=TrainingConfig)
    data_dir:str = f"./data/tokenized"
    ckpt_dir:str = f"./experiments/checkpoints"


