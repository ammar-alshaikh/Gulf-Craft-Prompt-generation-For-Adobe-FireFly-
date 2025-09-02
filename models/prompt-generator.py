#!/usr/bin/env python3
"""
Gulf Craft Prompt Generator - AI Model Interface
This script provides an interface for the fine-tuned AI model to generate yacht design prompts.
"""

import sys
import json
import logging
from typing import Dict, Any, Optional
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptGenerator:
    def __init__(self):
        """Initialize the prompt generator with the fine-tuned model."""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self, model_path: str = None):
        """
        Load the fine-tuned T5 model.
        
        Args:
            model_path: Path to the model files (if None, uses default path)
        """
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            
            if model_path is None:
                model_path = "./t5-model/gulf_craft_firefly_final"
            
            logger.info(f"Loading T5 model from: {model_path}")
            
            # Load T5 tokenizer and model
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            
            logger.info("T5 model loaded successfully")
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load T5 model: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_loaded = False
    
    def generate_prompt(self, user_input: str, max_length: int = 200) -> Dict[str, Any]:
        """
        Generate a yacht design prompt based on user input using T5 model.
        
        Args:
            user_input: The user's description or request
            max_length: Maximum length of generated prompt
            
        Returns:
            Dictionary containing the generated prompt and metadata
        """
        try:
            if not self.is_loaded:
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "prompt": None
                }
            
            # Prepare input for T5 (add prefix if your model was trained with one)
            # Adjust this based on how your model was fine-tuned
            input_text = f"generate yacht prompt: {user_input}"
            
            # Tokenize input
            inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate with T5
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output (remove input prefix if present)
            if generated_text.startswith("generate yacht prompt:"):
                generated_text = generated_text.replace("generate yacht prompt:", "").strip()
            
            return {
                "success": True,
                "prompt": generated_text,
                "input_length": len(user_input),
                "output_length": len(generated_text),
                "model_info": "Gulf Craft T5 Fine-tuned Prompt Generator"
            }
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "prompt": None
            }

# Global instance
prompt_generator = PromptGenerator()

def main():
    """Main function for testing the model directly."""
    if len(sys.argv) < 2:
        print("Usage: python prompt-generator.py 'your yacht description'")
        sys.exit(1)
    
    user_input = sys.argv[1]
    
    # Load model (you'll need to specify the path to your model)
    prompt_generator.load_model()
    
    # Generate prompt
    result = prompt_generator.generate_prompt(user_input)
    
    if result["success"]:
        print("Generated Prompt:")
        print(result["prompt"])
    else:
        print("Error:", result["error"])

if __name__ == "__main__":
    main() 