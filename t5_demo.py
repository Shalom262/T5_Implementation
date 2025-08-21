from transformers import T5Tokenizer, T5ForConditionalGeneration

def run_t5_translation(text):
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Preprocess the text
    preprocess_text = f"translate English to German: {text}"
    input_ids = tokenizer.encode(preprocess_text, return_tensors='pt')

    # Generate translation
    outputs = model.generate(input_ids, max_length=40, num_beams=4, early_stopping=True)

    # Decode and print the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Translated Text: {translated_text}")

if __name__ == "__main__":
    sample_text = "Hi this is Shalom here, how are you?"
    run_t5_translation(sample_text)
