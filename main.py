from data_analyzer import MvpDataAnalyzer, CompleteDataAnalyzer

if __name__ == "__main__":
    # Change this to 
    # model_name = input("Choose an LLM (openai, anthropic, llama): ").strip().lower()
    # to give the user the choice of which model to use - I only tested with anthropic, so I can't guarantee that it will work with the other models
    model_name = "anthropic" 
    
    # Change this to MvpDataAnalyzer() to use the MVP implementation
    analyzer = CompleteDataAnalyzer(model_name=model_name)

    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = analyzer.answerQuestion(messages, user_input)
        print("Bot:", response)
