from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, T5ForConditionalGeneration
from base_summarizer import BaseSummarizer
import torch

class T5QASummarizer(BaseSummarizer):
    def __init__(self, model_name="t5-base", 
                 qa_model_name="deepset/roberta-base-squad2", 
                 max_input_length=1024, max_output_length=350, device=None):
        super().__init__(model_name, max_input_length, max_output_length, device)
        # Initialize the summarization model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        # Initialize the Q&A model 
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(self.device)

    def summarize_text(self, input_text):
        inputs = self._tokenize_input(self.tokenizer, input_text)
        with torch.no_grad():
            summary_ids = self.model.generate(inputs['input_ids'], max_length=self.max_output_length, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def answer_question(self, input_text, question):
        input_text = f"question: {question} context: {input_text}"

        inputs = self.tokenizer(input_text, truncation=True, padding=True, max_length=self.max_input_length, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(inputs['input_ids'], max_length=self.max_output_length, num_beams=4, early_stopping=True)

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer