import os
import yaml
from datetime import datetime
import time
import math
import numpy as np
from pathlib import Path
import torch
from dataclasses import asdict
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoTokenizer
from .config import SelfAttentionConfig, TransformerConfig,DecoderLMConfig, RunConfig, TrainingConfig
from .models.causal_llm import TransformerLM
from .data import get_batch, SequenceDataset
from .scheduler import get_cosine_schedule_with_warmup
from .decoding import TextDecoder
from tqdm.auto import tqdm
import wandb

class Trainer:

    def __init__(
        self,
        model: TransformerLM,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        train_dataloader:callable,
        val_dataloader:DataLoader,
        decoder: TextDecoder,
        config: RunConfig):

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.decoder = decoder
        self.config = config

        # USE - batch_size = 64, context_length = 256, num_iterations = 20000

        self.patience_threshold = config.train.patience_threshold
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.resume_from_checkpoint = config.resume_from_checkpoint
        self.dataset_name = config.data_dir.split('/')[-1]
        self.run_id = None
        self._setup_experiment()

    def _setup_experiment(self):

        if self.resume_from_checkpoint:
            
            ckpt_path = self.resume_from_checkpoint
            self._load_checkpoint(ckpt_path)

            wandb.init(
                project="mintransformer",
                id=self.run_id,
                resume="must",
                mode=self.config.wandb_mode, 
                config={**asdict(self.config.train),
                        **asdict(self.config.model),
                        "dataset":self.dataset_name,
                        "total_params": sum(p.numel() for p in self.model.parameters())},
            )
            
            resume = True # Flag that indicates that we are resuming from an existign experiment
            
        else:

            self.global_step = 0
            self.best_perplexity = float('inf')
            self.patience_counter = 0
            self.best_iteration = 0
            wandb.init(
                project="mintransformer",
                name = f"{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                mode=self.config.wandb_mode, 
                config={**asdict(self.config.train),
                        **asdict(self.config.model),
                        "dataset":self.dataset_name,
                        "total_params": sum(p.numel() for p in self.model.parameters()),
                        "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)})

            self.run_id = wandb.run.id
            resume = False
        
        self.exp_dir = Path(f"./experiments/{self.dataset_name}/{self.run_id}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        (self.exp_dir / "samples").mkdir(exist_ok=True)

        if not resume:
            config_dict = {
            'wandb_info': {
                'run_id': self.run_id,
                'run_name': wandb.run.name,
                'url': wandb.run.get_url(),
                'project': wandb.run.project,
                'timestamp': datetime.now().isoformat()},
            'dataset': self.dataset_name,
            'training_config': asdict(self.config.train),
            'model_config': asdict(self.config.model),
            'generation_config': {
                'temperature': self.decoder.temperature,
                'top_p': self.decoder.top_p},
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)}}
        
        with open(self.exp_dir / "config.yaml", 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
    def _evaluate(self) -> float:

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):

                input_ids = batch[0].to(self.device)
                model_val_logits = self.model(input_ids)
                target_ids = batch[1].to(self.device)
                        
                val_loss = cross_entropy(
                        model_val_logits.view(-1, model_val_logits.size(-1)), 
                        target_ids.view(-1),reduction='sum'
                    )
                total_loss += val_loss.item()
                total_tokens += target_ids.numel()  
            avg_loss_per_token = total_loss/total_tokens      
            results = {"avg_perplexity": math.exp(avg_loss_per_token), "avg_val_loss": avg_loss_per_token}

        return results

    def _save_checkpoint(self):

        checkpoint_path = self.exp_dir / "checkpoints" 
        print(f" Saving model ckpt to : {checkpoint_path}")
        checkpoint = {
            "global_step": self.global_step,
            "best_perplexity": self.best_perplexity,
            "best_iteration": self.best_iteration,
            "patience_counter": self.patience_counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "run_id": self.run_id,
        }

        torch.save(
            checkpoint, f"{checkpoint_path}/best_model.pth"
        )

    def _load_checkpoint(self, checkpoint_path: str):
        
        if os.path.exists(checkpoint_path):
            print(f" Loading ckpt from : {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if not self.config.train.reset_scheduler_on_load:
                print(f"Scheduler state loaded from checkpoint.")
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                print(f"Scheduler state NOT loaded. A new scheduler will be used.")

            self.global_step = checkpoint["global_step"]+1
            self.best_perplexity = checkpoint["best_perplexity"]
            self.run_id = checkpoint["run_id"]
            self.best_iteration = checkpoint["best_iteration"]
            self.patience_counter = checkpoint["patience_counter"]

            print(f"Resuming training from iteration {self.global_step}")
        
        else:
            
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")


    def _load_best_model(self):
        checkpoint_path = self.exp_dir / "checkpoints"/"best_model.pth" 
        if os.path.exists(checkpoint_path):
            print(f" Loading ckpt from : {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model weights from iteration {checkpoint['global_step']}")
            print(f"Best validation perplexity: {checkpoint['best_perplexity']:.4f}")
            self.global_step = checkpoint["global_step"]+1
            return checkpoint
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    def _generate_and_log_samples(self, step, max_tokens = 200):

        self.model.eval()
        prompts = ["Once upon a time",
                   "There once was a little boy",
                   "In a magical land far, far away"]
        
        sampled_text = f"=== Step {step} ===\n"
        sampled_text += f"Temperature: {self.decoder.temperature}, Top-p: {self.decoder.top_p}\n\n"
        log_samples = []

        for prompt in prompts:
            generated = self.decoder.generate(self.model, prompt, max_tokens)
            sample_text = f"Prompt: {prompt}\nGenerated: {generated}\n\n"
            sampled_text += sample_text
            log_samples.append([step, prompt, generated])

        
        table = wandb.Table(
        columns=["Step", "Prompt", "Generated"],
        data=log_samples)
        wandb.log({"generated_samples": table}, step = step)
    
        if step % 2500 == 0 or step == self.config.train.num_iterations:
            sample_file = self.exp_dir / "samples" / f"step_{step}.txt"
            with open(sample_file, 'w') as f:
                f.write(sampled_text)
        
        self.model.train()
        
    def train(self):

        print(f" Starting training...")
        early_stopped = False

        for i in range(self.config.train.num_iterations):
            self.model.train()
            batch = next(self.train_dataloader)
            step_start = time.time()
            self.optimizer.zero_grad()
            model_logits = self.model(batch[0])
            batch_loss = cross_entropy(model_logits.view(-1, self.config.model.vocab_size), batch[1].view(-1))
            batch_loss.backward()
            total_grad_norm  = clip_grad_norm_(self.model.parameters(),self.config.train.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            step_time = time.time() - step_start
            num_tokens = batch[0].numel()
            tokens_per_sec = num_tokens / step_time if step_time > 0 else 0
            self.global_step += 1
              
            wandb.log({"train/batch_loss": batch_loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "train/perplexity": math.exp(batch_loss.item()),
                        "gradients/norm_before_clip":total_grad_norm,
                        "memory/gpu_gb":torch.cuda.memory_allocated()/1024**3,
                        "throughput/step_time":step_time,
                        "throughput/tokens_per_sec":tokens_per_sec}, step=self.global_step)

            print(f"Iteration-{self.global_step}: train_loss={batch_loss.item()}")

            if self.global_step % self.config.train.eval_every_n_steps == 0:

                print(f"--- Intra-epoch eval at step {self.global_step}---")

                eval_results = self._evaluate()
                
                wandb.log({"val/avg_perplexity": eval_results["avg_perplexity"],
                        "val/avg_loss_per_token": eval_results["avg_val_loss"]},
                    step=self.global_step)

                print(f"Iteration-{self.global_step}: train_loss={batch_loss.item()},\
                        val_loss={eval_results['avg_val_loss']}")
                
                self._generate_and_log_samples(self.global_step)
                
                if eval_results['avg_perplexity'] < self.best_perplexity:
                    perplexity_improvement = eval_results['avg_perplexity'] - self.best_perplexity
                    self.best_perplexity  = eval_results['avg_perplexity']
                    self.best_iteration = self.global_step
                    self.patience_counter = 0
                    
                    print(f"New best perpelxity: {self.best_perplexity:.4f}.")
                    self._save_checkpoint()
                    
                    wandb.log({"early_stopping/best_val_perplexity": self.best_perplexity,
                            "early_stopping/best_iteration": self.best_iteration,
                            "early_stopping/patience_counter": self.patience_counter,
                            "early_stopping/improvement": perplexity_improvement},
                        step=self.global_step)

                else:

                    self.patience_counter += 1
                    print(f"No improvement. Patience: {self.patience_counter}/{self.patience_threshold}")
                                                           
                    wandb.log({"early_stopping/best_val_perplexity": self.best_perplexity,
                            "early_stopping/best_iteration": self.best_iteration,
                            "early_stopping/patience_counter": self.patience_counter},
                        step=self.global_step)
                    
                    if self.patience_counter >= self.patience_threshold:
                        print(f" Early stopping triggered at iteration {self.global_step}")
                        print(f" Best validation perplexity of {self.best_perplexity} at iteration {self.best_iteration}")
                        early_stopped = True
                        break  
        
        print("Training complete.")
        print("\n" + "=" * 60)
        _ = self._load_best_model()
        print("Performing Final evaluation")
        final_eval_results = self._evaluate()
        final_perplexity = final_eval_results["avg_perplexity"]
        final_avg_val_loss_per_token = final_eval_results["avg_val_loss"]

        print(f"\nFinal Model Performance:")
        print(f" Validation Perplexity: {final_perplexity:.4f}")
        print(f" Validation Loss: {final_avg_val_loss_per_token:.4f}")
        print(f" From Iteration: {self.best_iteration}")

        perplexity_diff = abs(final_perplexity - self.best_perplexity)
        if perplexity_diff > 0.01: # check 
            print(f"WARNING: Final perplexity ({final_perplexity:.4f}) differs from "
                f"saved best perplexity ({self.best_perplexity:.4f})")
        else:
            print(f"Final perplexity matches saved best checkpoint")
            wandb.run.summary["final/val_perplexity"] = final_perplexity
            wandb.run.summary["final/val_loss"] = final_avg_val_loss_per_token 
            wandb.run.summary["best_iteration"] = self.best_iteration
            wandb.run.summary["total_iterations"] = self.global_step


        if early_stopped:
            print(f"Training stopped early at iteration {self.global_step}")
            wandb.run.summary["early_stopped"] = True
            wandb.run.summary["stopped_at_iteration"] = self.global_step
        else:
            print("Training completed all iterations")
            wandb.run.summary["early_stopped"] = False
           
        print("=" * 60)
        wandb.finish()

        return self.model, final_eval_results
    
def train_model(args):

    ## Loading data from disk

    try:
       data_train = np.load(os.path.join(args.data_dir,'train.npy'),mmap_mode='r')
    except FileNotFoundError as e:
        raise e

    print("Mapped data from disk")

    ## Creating the configs
    train_config = TrainingConfig(learning_rate=args.lr,
                                  weight_decay=args.weight_decay,
                                  beta1 = args.beta1,
                                  beta2 = args.beta2,
                                  num_iterations=args.num_iters,
                                  patience_threshold=args.patience_threshold,
                                  batch_size = args.batch_size,
                                  reset_scheduler_on_load = args.reset_sched_on_load,
                                  eval_every_n_steps=args.eval_n_steps,
                                  min_lr = 0.1*args.lr,
                                  num_warmup_steps = int(args.frac_warmup_steps * args.num_iters),
                                  num_cosine_steps = int(args.frac_cosine_steps * args.num_iters))
    
    model_config = DecoderLMConfig(d_model=args.d_model,
                                   vocab_size = args.vocab_size,
                                   context_length = args.ctx_len,
                                   n_layers = args.n_layers,
                                   pos_embedding_type=args.pos_embed,
                                   dropout = args.dropout,
                                   transformer=TransformerConfig(norm_position = args.norm_pos,
                                                                 norm_type = args.norm_type,
                                                                 attn = SelfAttentionConfig(
                                                                     n_heads=args.n_heads, 
                                                                     dropout_attn=args.dropout_attn))) 

    config = RunConfig(model=model_config,
                       train=train_config,
                       data_dir=args.data_dir,
                       wandb_mode=args.wandb_mode)
    
    # Initializing the model
    model = TransformerLM(model_config, device="cuda:0")

    train_loader = get_batch(data_train, 
                             batch_size=train_config.batch_size,
                             context_length=model_config.context_length, 
                             device = "cuda:0")
    
    # Note that setting stride = ctx_len will produce validation context windows that are non-overlapping
    val_dataset = SequenceDataset(file_path = os.path.join(args.data_dir,'validation.npy'),
                        context_length=model_config.context_length, stride = model_config.context_length)

    val_loader = DataLoader(
        val_dataset ,
        batch_size=train_config.batch_size,
        shuffle=False, 
        pin_memory=torch.cuda.is_available(),
        drop_last=False)

    
    if config.train.weight_decay is not None:

        optimizer = AdamW(
            model.parameters(),
            weight_decay=config.train.weight_decay,
            lr=config.train.learning_rate,
            betas = (config.train.beta1, config.train.beta2)
        )
    
    else:

        optimizer = AdamW(
            model.parameters(),
            weight_decay=0.0,        # reduces to standard adam
            lr=config.train.learning_rate,
            betas = (config.train.beta1, config.train.beta2))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    decoder = TextDecoder(tokenizer = tokenizer, temperature = args.temperature,
                           top_p = args.top_p, device = "cuda:0")

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   max_lr = train_config.learning_rate,
                                                   min_lr = train_config.min_lr,
                                                   num_warmup_steps = train_config.num_warmup_steps,
                                                   num_cosine_steps = train_config.num_cosine_steps)

    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        decoder = decoder,
        config=config)

    final_model, final_eval_results = trainer.train()

    return final_model, final_eval_results