from typing import Dict, List, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

class EndpointHandler():
    def __init__(self, path="bert2gpt"):

        encoder = "bert-base-uncased"
        decoder = "aubmindlab/aragpt2-base"
        self.encoder_max_length=512
        self.decoder_max_length=512
        self.num_beams = 5
        self.model =AutoModelForSeq2SeqLM.from_pretrained(path)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder)
       # self.pipeline = pipeline("text-classification",model=path)

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str`)
            date (:obj: `str`)
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        # get inputs
        inputs = data.pop("inputs",data)
        # parameters = data.get("parameters", {})

        # tokenize the input
        input_ids = self.encoder_tokenizer(inputs, return_tensors="pt").input_ids
        # run the model
        # logits = self.model.generate(input_ids, **parameters)
        outputs = self.model.generate(input_ids,max_length = len(inputs.split())+1,length_penalty= 1.1,num_beams=5,no_repeat_ngram_size=2,do_sample = True,top_k=50)
        return  {"generated_text": self.decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)}