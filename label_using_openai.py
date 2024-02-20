import openai
import time
import processing as pre

openai.api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
# Function to label messages using OpenAI's API
def label_messages_with_chatgpt(messages):
    # Batch messages together
    batch_size = 5
    batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
    generated_labels = []
    for batch in batches:
        # Construct prompt for batch
        prompt = "\n".join([f"Label this message: {message}. as one of {labels}" for message in batch])
        # Call OpenAI API
        response = openai.chat.completions.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=len(batch),
            stop=None,
            temperature=0.7)
        # Extract generated label from the first choice
        for choice in response.choices:
            # Filter out labels from the choice text
            extracted_labels = [label.lower() for label in labels if label.lower() in choice.text.lower()]
            if extracted_labels:
                generated_labels.append(extracted_labels[0])
            else:
                generated_labels.append("Other")
            break
        #delay to prevent exceeding rate limits
        time.sleep(10)
    return generated_labels

if __name__ == '__main__':
    # Predefined list of labels
    labels = ["CPU usage", "Response time", "Throughput", "Disk Input/Output", "Memory usage"]
    df = pre.read_csv_files("10per_data.csv")
    df = pre.preprocessing(df, "message")
    df['label'] = label_messages_with_chatgpt(df['message_cleaned'])
    df.to_csv('10per_labeled.csv', index=False)   