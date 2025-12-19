"""LLM Policy wrapper for code generation agent."""

import re
import torch
from typing import Dict, Any, Optional, List, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

from ..utils.config import Config
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class GenerationOutput:
    """Output from model generation."""
    text: str
    token_ids: List[int]
    log_probs: Optional[torch.Tensor] = None
    finished: bool = True


class LLMPolicy:
    """LLM-based policy for code generation.
    
    Wraps a HuggingFace model with:
    - 4-bit quantization for M4 Mac
    - LoRA for efficient fine-tuning
    - Tool-use aware generation
    - Streaming support for real-time UI
    """
    
    def __init__(
        self,
        config: Config,
        device: Optional[str] = None
    ):
        """Initialize policy.
        
        Args:
            config: Configuration object
            device: Device to use (auto-detected if None)
        """
        self.config = config
        self.model_config = config.model
        
        # Determine device
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Will be set in load()
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Conversation history for multi-turn
        self.conversation_history: List[Dict[str, str]] = []
    
    def load(self):
        """Load model and tokenizer."""
        model_name = self.model_config.name
        logger.info(f"Loading model: {model_name}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        quant_config = None
        if self.model_config.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif self.model_config.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.float16,
        }
        
        if quant_config and self.device != "mps":
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            # For MPS or no quantization
            model_kwargs["device_map"] = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
        
        # Setup generation config
        gen_config = self.model_config.generation
        self.generation_config = GenerationConfig(
            max_new_tokens=gen_config.get('max_new_tokens', 2048),
            temperature=gen_config.get('temperature', 0.7),
            top_p=gen_config.get('top_p', 0.95),
            do_sample=gen_config.get('do_sample', True),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
    
    def setup_lora(self):
        """Setup LoRA for training."""
        if not self.config.training.lora.enabled:
            logger.info("LoRA disabled, skipping setup")
            return
        
        lora_config = self.config.training.lora
        
        # Prepare model for k-bit training if quantized (not on MPS)
        if self.model_config.quantization in ["4bit", "8bit"] and self.device != "mps":
            logger.info("Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Create LoRA config
        logger.info(f"Creating LoRA config: r={lora_config.r}, alpha={lora_config.alpha}")
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        logger.info("Applying LoRA to model...")
        self.model = get_peft_model(self.model, peft_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LoRA applied: {trainable_params/1e6:.2f}M trainable / {total_params/1e6:.2f}M total ({100*trainable_params/total_params:.2f}%)")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_log_probs: bool = False
    ) -> GenerationOutput:
        """Generate response from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_new_tokens: Override max tokens
            temperature: Override temperature
            return_log_probs: Whether to return log probabilities
            
        Returns:
            GenerationOutput with generated text
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_context_length - (max_new_tokens or 2048)
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Update generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.generation_config.max_new_tokens,
            temperature=temperature or self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            do_sample=self.generation_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=return_log_probs,
            return_dict_in_generate=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compute log probs if requested
        log_probs = None
        if return_log_probs and hasattr(outputs, 'scores') and len(outputs.scores) > 0:
            # Stack scores: (seq_len, batch_size, vocab_size) -> (seq_len, vocab_size)
            scores = torch.stack(outputs.scores, dim=0).squeeze(1)  # (seq_len, vocab_size)
            all_log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
            
            # Gather log probs for the actual generated tokens
            # generated_ids: (seq_len,)
            token_log_probs = all_log_probs.gather(1, generated_ids.unsqueeze(-1)).squeeze(-1)
            log_probs = token_log_probs  # (seq_len,) tensor of log probs for each token
        
        return GenerationOutput(
            text=generated_text,
            token_ids=generated_ids.tolist(),
            log_probs=log_probs,
            finished=True
        )
    
    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        callback: Optional[callable] = None
    ) -> Generator[str, None, None]:
        """Generate response with streaming for real-time display.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            callback: Optional callback for each token
            
        Yields:
            Generated tokens one at a time
        """
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_context_length - 2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generate in background thread
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "generation_config": self.generation_config
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens
        full_text = ""
        for token in streamer:
            full_text += token
            if callback:
                callback(token)
            yield token
        
        thread.join()
    
    def compute_log_prob(
        self,
        prompt: str,
        response: str,
        require_grad: bool = False
    ) -> torch.Tensor:
        """Compute log probability of response given prompt.
        
        Args:
            prompt: Input prompt
            response: Generated response
            require_grad: Whether to track gradients (True for training updates)
            
        Returns:
            Log probability tensor
        """
        # Concatenate prompt and response
        full_text = prompt + response
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        prompt_len = prompt_ids.shape[1]
        
        if require_grad:
            # With gradients for policy update
            outputs = self.model(**inputs)
            logits = outputs.logits
        else:
            # Without gradients for inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
        
        # Get log probs for response tokens only
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        response_log_probs = []
        for i in range(prompt_len, inputs['input_ids'].shape[1] - 1):
            token_id = inputs['input_ids'][0, i + 1]
            response_log_probs.append(log_probs[0, i, token_id])
        
        if response_log_probs:
            return torch.stack(response_log_probs)
        return torch.tensor([0.0], device=self.model.device, requires_grad=require_grad)
    
    def save(self, path: str):
        """Save model weights.
        
        Args:
            path: Path to save to
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint_path = Path(path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load LoRA weights
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(checkpoint_path, adapter_name="default")
        else:
            # Load full model
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path
            )
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def reset_conversation(self):
        """Reset conversation history for new episode."""
        self.conversation_history = []
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history.
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_conversation_prompt(self) -> str:
        """Get full conversation as prompt.
        
        Returns:
            Formatted conversation string
        """
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                self.conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Fallback formatting
        parts = []
        for msg in self.conversation_history:
            role = msg['role'].upper()
            parts.append(f"{role}: {msg['content']}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)
