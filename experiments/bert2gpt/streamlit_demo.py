import streamlit as st # pip install streamlit==0.82.0
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; direction: rtl;">{}</div>"""
MODEL_PATH = "bert2gpt"

 
@st.cache_resource()
def load_model(model_path, encoder = "bert-base-uncased",decoder = "aubmindlab/aragpt2-base"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder)
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder)

    return model, encoder_tokenizer,decoder_tokenizer

def render(output):
    html = f"<div><p>{output}</p></div>"
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

def main():
    """Simple Translation Streamlit App"""
    
    st.title("EN to AR Translator")

    model, encoder_tokenizer,decoder_tokenizer = load_model(MODEL_PATH)
    input_txt = st.text_area("", placeholder="English Text")
    input_ids = encoder_tokenizer(input_txt, return_tensors="pt").input_ids

    if st.button("translate"):
        if input_txt == "":
            st.warning('Please **enter text** for translation')
        else:
            outputs = model.generate(input_ids,max_length = len(input_txt.split())+1,length_penalty= 1.1,num_beams=5,no_repeat_ngram_size=2,do_sample = True,top_k=50)
            print(input_ids.shape)
            render(decoder_tokenizer.decode(outputs[0], skip_special_tokens=True))
           

if __name__ == '__main__':
    main()