import argparse
from min_llm.train import train_model

def main():

    parser = argparse.ArgumentParser(description="Train causalLM on TinyStories")

    parser.add_argument( "--data_dir", type=str, 
                        default="../data/tokenized/tinystories", help="Directory containing data .npy files.")
    
    parser.add_argument( "--ckpt_dir", type=str, 
                        default="../experiments/tinystories/checkpoints",
                        help="Directory to store model checkpoints.")
    
    parser.add_argument( "--lr", type=float, 
                        default=0.001, 
                        help="max learning rate")
    
    parser.add_argument( "--weight_decay", type=float, 
                        default=0.01, 
                        help="weight decay rate for adam")
    
    parser.add_argument( "--beta1", type=float, 
                        default=0.90, 
                        help="beta1 for adam")
    
    parser.add_argument( "--beta2", type=float, 
                        default=0.95, 
                        help="beta2 for adam")
    
    parser.add_argument( "--num_iters", type=int, 
                        default=2500, 
                        help="number of training iterations")
    
    parser.add_argument( "--patience_threshold", type=int, 
                        default=3, 
                        help="patience for early stopping")
    
    parser.add_argument( "--batch_size", type=int,
                        default=64, 
                        help="minibatch size for training")
    
    parser.add_argument( "--resume_from_ckpt", type=str, 
                        default=None, 
                        help="ckpt path to load for resuming training")
    
    parser.add_argument( "--reset_sched_on_load", action='store_true',
                        help="flag that specifies whether to reset scheduler state")
    
    parser.add_argument( "--eval_n_steps", type = int,
                        default=500,
                        help="frequency at which to evaluate")
    
    parser.add_argument( "--frac_warmup_steps", type=float, 
                        default=0.05, 
                        help="fraction of traning steps to use for warmup")
    
    parser.add_argument( "--frac_cosine_steps", type=float, 
                        default=0.95, 
                        help="fraction of traning steps to use for warmup + cosine schedule")
    
    parser.add_argument("--tokenizer_name",type=str, 
                        default="openai-community/gpt2", 
                        help="Name of the Hugging Face tokenizer to use.")
    
    parser.add_argument("--temperature",type=float, 
                    default=1.0, 
                    help="Temperature for sotfmax sampling of text")
    
    parser.add_argument("--top_p",type=float, 
                default=0.1, 
                help="cdf threshold for nucleus sampling")
    
    parser.add_argument( "--d_model", type=int, 
                        default=512, 
                        help="model dimensionality")
    
    parser.add_argument( "--vocab_size", type=int, 
                        default=50257, # standard for gpt-2 tokenizer
                        help="size of vocabulary (depends on tokenizer)")
    
    parser.add_argument( "--ctx_len", type=int, 
                        default=256, 
                        help="maximum context length")
    
    parser.add_argument( "--n_layers", type=int, 
                        default=4, 
                        help="number of layers")
    
    parser.add_argument( "--n_heads", type=int, 
                        default=16, 
                        help="number of attention heads")
    
    parser.add_argument( "--pos_embed", type=str, 
                        default="rope", 
                        help="type of position embedding")
    
    parser.add_argument( "--dropout", type=float, 
                        default=0.1, 
                        help="dropout")
    
    parser.add_argument( "--dropout_attn", type=float, 
                        default=0.1, 
                        help="dropout for self attention")
    
    parser.add_argument( "--norm_type", type=str, 
                        default="rms", 
                        help="type of nomr layer")
    
    parser.add_argument( "--norm_pos", type=str, 
                        default="pre", 
                        help="position of norm layer")
    

    args = parser.parse_args()
    final_model, final_eval_results = train_model(args)
    return final_model, final_eval_results


if __name__ == "__main__":

    final_model, final_eval_results = main()

    
