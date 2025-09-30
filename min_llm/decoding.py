import torch
from torch.distributions.categorical import Categorical
from .models.causal_llm import TransformerLM
from .functional import softmax
class Decoder:

    def __init__(self, tokenizer, temperature = 1.0, top_p = None, device = None):

        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

    def _truncate(self, logits):

        top_p_logits = torch.empty_like(logits)
        top_p_logits[:] = float('-inf')

        prob_dist = Categorical(logits = logits)
        idxs = torch.argsort(logits)
        cum_probs = torch.cumsum(prob_dist.probs[idxs], dim = 0)
        idxs_mask = torch.where(cum_probs >= 1.0 - self.top_p, True, False)

        top_p_logits[idxs[idxs_mask]] = logits[idxs[idxs_mask]]
        prob_dist = Categorical(logits = top_p_logits)

        return prob_dist


    def decode(self, model:TransformerLM,
               input_prompt:str, max_tokens = 200):

        input_ids = torch.tensor(self.tokenizer.encode(input_prompt)).long().to(self.device)
        outputs = []
        with torch.no_grad():
            for i in range(max_tokens):
                out_logits = torch.squeeze(model(input_ids.view(1,-1)))
                temp_scaled_logits = out_logits[-1,:] * (1.0 / self.temperature)
                
                if self.top_p is not None:
                    prob_dist = self._truncate(logits = temp_scaled_logits)
                else:
                    prob_dist = Categorical(logits = temp_scaled_logits)
                
                output_token = prob_dist.sample()
                if output_token == self.tokenizer.eos_token_id:
                    outputs.append(self.tokenizer.decode(output_token))
                    return ''.join(outputs)

                outputs.append(self.tokenizer.decode(output_token))
                if i != max_tokens - 1:
                    input_ids = torch.concat([input_ids, output_token.view(1).long()])
        return ''.join(outputs)
