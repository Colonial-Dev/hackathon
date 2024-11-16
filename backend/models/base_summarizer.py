# base_summarizer.py
import torch

class BaseSummarizer:
    """
    Contains common functionaty for text summarizers, which inherit from this class and use specific models (such as BARTM Llama3, etc).
    """
    def __init__(self, model_name, max_input_length=1024, max_output_length=350, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def _tokenize_input(self, tokenizer, input_text):
        return tokenizer(input_text, return_tensors="pt", max_length=self.max_input_length, truncation=True, padding="longest").to(self.device)

    def _chunk_text(self, input_text):
        """
        Split the input text into chunks of a max length, ensuring it's within BART's token limit.

        TODO: ADD OVERLAP BETWEEN CHUNKS OF TEXT TO AVOID LOSING CONTEXT?

        Args:
            input_text (str): The text to be split into chunks.

        Returns:
            list: A list of text chunks.
        """
        inputs = self._tokenize_input(input_text)
        num_chunks = (len(inputs["input_ids"][0]) // self.max_input_length) + 1
        return [input_text[i * self.max_input_length:(i + 1) * self.max_input_length] for i in range(num_chunks)]
