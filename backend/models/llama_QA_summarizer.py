from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
from base_summarizer import BaseSummarizer
import torch

class LlamaQASummarizer(BaseSummarizer):
    def __init__(self, model_name="raaec/Meta-Llama-3.1-8B-Instruct-Summarizer", 
                 qa_model_name="deepset/roberta-base-squad2", 
                 max_input_length=1024, max_output_length=350, device=None):
        super().__init__(model_name, max_input_length, max_output_length, device)
        # Initialize the summarization model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Initialize the Q&A model
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(self.device)

    def summarize_text(self, input_text):
        inputs = self._tokenize_input(self.tokenizer, input_text)
        with torch.no_grad():
            summary_ids = self.model.generate(inputs['input_ids'], max_length=self.max_output_length, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def answer_question(self, input_text, question):
        inputs = self.qa_tokenizer(input_text, truncation=True, padding=True, max_length=self.max_input_length, return_tensors="pt").to(self.device)
        question_inputs = self.qa_tokenizer(question, truncation=True, padding=True, max_length=64, return_tensors="pt").to(self.device)
        
        inputs = {key: torch.cat([question_inputs[key], inputs[key]], dim=-1) for key in inputs}
        
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits
            
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)

        # Extract the tokens for the answer and decode
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        return self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
