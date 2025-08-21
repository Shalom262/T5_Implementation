from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

def load_model():
    model_name = 't5-base'
    model_path = f'./{model_name}'

    if not os.path.exists(model_path):
        print("Downloading model...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
    else:
        print("Loading model from local path...")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

def perform_task(prefix, text, tokenizer, model):
    input_text = f"{prefix}: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    tokenizer, model = load_model()

    services = {
        "1": "Text Summarization",
        "2": "Translation",
        "3": "Question Answering",
        "4": "Text Classification (Sentiment)",
        "5": "Grammar Correction",
        "6": "Paraphrasing"
    }

    while True:
        print("\nPlease select a service:")
        for key, value in services.items():
            print(f"{key}: {value}")

        choice = input("Enter the number of the service you want to use: ")

        if choice not in services:
            print("Invalid choice. Please try again.")
            continue

        service_name = services[choice]
        
        while True:
            print(f"\nYou have selected: {service_name}\n")

            if choice == "1": # Text Summarization
                text = input("Enter the text to summarize: ")
                summary = perform_task("summarize", text, tokenizer, model)
                print(f"\nSummary:\n{summary}")

            elif choice == "2": # Translation
                supported_languages = ["German", "French", "Romanian"]
                print(f"Supported target languages: {', '.join(supported_languages)}")
                target_lang = input("Enter the target language: ")
                while target_lang not in supported_languages:
                    print("Invalid target language.")
                    target_lang = input("Enter the target language: ")
                text = input("Enter the English text to translate: ")
                translation = perform_task(f"translate English to {target_lang}", text, tokenizer, model)
                print(f"\nTranslation:\n{translation}")

            elif choice == "3": # Question Answering
                context = input("Enter the context: ")
                question = input("Enter the question: ")
                qa_input = f"question: {question} context: {context}"
                answer = perform_task("", qa_input, tokenizer, model)
                print(f"\nAnswer:\n{answer}")

            elif choice == "4": # Text Classification
                text = input("Enter the text to classify for sentiment (positive/negative): ")
                classification = perform_task("sst2 sentence", text, tokenizer, model)
                print(f"\nSentiment:\n{classification}")

            elif choice == "5": # Grammar Correction
                text = input("Enter the text for grammar correction: ")
                correction = perform_task("grammatically correct:", text, tokenizer, model)
                print(f"\nCorrected Text:\n{correction}")

            elif choice == "6": # Paraphrasing
                text = input("Enter the text to paraphrase: ")
                paraphrase = perform_task("rewrite this sentence:", text, tokenizer, model)
                print(f"\nParaphrase:\n{paraphrase}")

            next_action = input("\nWhat would you like to do next?\n1. Continue with this feature\n2. Go back to the main menu\n3. Exit\nEnter your choice (1/2/3): ")
            if next_action == '1':
                continue
            elif next_action == '2':
                break
            elif next_action == '3':
                print("Exiting the program. Goodbye!")
                return
            else:
                print("Invalid choice. Returning to the main menu.")
                break

if __name__ == "__main__":
    main()
