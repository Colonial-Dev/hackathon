from transformers import BartForConditionalGeneration, BartTokenizer
from base_summarizer import BaseSummarizer
import torch

class BARTSummarizer(BaseSummarizer):
    def __init__(self, model_name="facebook/bart-large-cnn", max_input_length=1024, max_output_length=350, device=None):
        super().__init__(model_name, max_input_length, max_output_length, device)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def summarize_text(self, input_text):
        inputs = self._tokenize_input(self.tokenizer, input_text)
        with torch.no_grad():
            summary_ids = self.model.generate(inputs['input_ids'], max_length=self.max_output_length, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
