from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.optim import Optimizer

def get_cosine_schedule_with_warmup(optimizer: Optimizer, 
                                    max_lr: float,
                                    min_lr: float,
                                    num_warmup_steps: int,
                                    num_cosine_steps: int):
        
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, 
                              end_factor=1.0, total_iters=num_warmup_steps)

        cosine_scheduler = CosineAnnealingLR(optimizer,
                                       T_max = num_cosine_steps - num_warmup_steps,
                                       eta_min=min_lr)
        
        constant_scheduler = ConstantLR(optimizer, factor = min_lr/max_lr,  
                                total_iters=float('inf'))
        
        scheduler = SequentialLR(optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
        milestones=[num_warmup_steps, num_cosine_steps])

        return scheduler
