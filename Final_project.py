# %%
import csv
from collections import defaultdict
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Function to preprocess the text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove punctuations
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove blank space tokens
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

# Function to create inverted index
def create_inverted_index(file_path):
    inverted_index = defaultdict(set)  # Use set instead of list to automatically remove duplicates
    
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            line_number = int(row['Line Number'])
            line_data = row['Data']
            tokens = preprocess_text(line_data)
            for token in set(tokens):  # Use set to remove duplicate tokens in the same line
                inverted_index[token].add(line_number)
    
    return inverted_index

# Function to write inverted index to CSV file
def write_index_to_csv(index, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Token', 'Line Numbers']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for token, line_numbers in index.items():
            writer.writerow({'Token': token, 'Line Numbers': ', '.join(map(str, sorted(line_numbers)))})


def one():
    file_path = 'formatted_news_data.csv'

    # Create inverted index
    inverted_index = create_inverted_index(file_path)

    # Replace 'index.csv' with your desired output file path
    output_file = 'index.csv'

    # Write inverted index to CSV file
    write_index_to_csv(inverted_index, output_file)

    print(f"Inverted index has been written to {output_file}")


# %%
#2
import string
import pandas as pd

# Function to read inverted index from CSV file
def read_inverted_index(file_path):
    inverted_index = {}
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        token = row['Token']
        line_numbers = [int(num) for num in row['Line Numbers'].split(',')]
        inverted_index[token] = line_numbers
    return inverted_index

# Function to preprocess text and get tokens
def preprocess_text(text):
    stopwords = [
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
        "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ]
    
    tokens = text.lower().split()  
    tokens = [token.strip(string.punctuation) for token in tokens]  
    tokens = [token for token in tokens if token and token not in stopwords]  
    return tokens

# Function to find line numbers for tokens from the inverted index
def find_line_numbers(tokens, inverted_index):
    line_numbers = []
    for token in tokens:
        if token in inverted_index:
            line_numbers.extend(inverted_index[token])
    return line_numbers

# Function to write line numbers list to CSV file
def write_line_numbers_to_csv(line_numbers, output_file):
    with open(output_file, 'w') as file:
        for line_number in line_numbers:
            file.write(f"{line_number}\n")

# Combined function to perform all tasks
def perform_all(user_input):
    # Read the inverted index CSV file
    file_path = 'index.csv'
    output_file = 'formatted_line_numbers.csv'
    inverted_index = read_inverted_index(file_path)
    
    # Take input from the user
    # user_input = input("Enter a sentence: ")
    
    # Preprocess the input and get tokens
    tokens = preprocess_text(user_input)

    # Find line numbers for tokens from the inverted index
    line_numbers = find_line_numbers(tokens, inverted_index)

    # Write line numbers list to CSV file
    write_line_numbers_to_csv(line_numbers, output_file)

    print(f"Line numbers have been written to {output_file}")
    
    # Return both the tokens and line numbers
    return tokens, line_numbers,user_input




# %%
#3
import pandas as pd

# Define the file path for the CSV file containing line numbers and data
csv_file_path = 'formatted_news_data.csv'  # Replace 'formatted_news_data.csv' with your CSV file path

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Function to retrieve data corresponding to line numbers
def three(line_numbers):
    # Initialize an empty string to store the concatenated data
    context = ""

    # Convert line numbers to a set
    line_num = set(line_numbers)

    # Loop through the set of line numbers and retrieve the corresponding data
    for line_number in line_num:
        # Check if the line number is within the range of the DataFrame
        if line_number in df['Line Number'].values:
            # Retrieve the data corresponding to the line number
            line_data = df[df['Line Number'] == line_number]['Data'].values[0]
            # Concatenate the line data to the context string
            context += line_data + '\n'
        else:
            print(f"No data found for line number {line_number}")

    # Return the concatenated data
    return context




# %%
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def four(context, question):
    # Load model and tokenizer
    model_name = "deepset/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get predictions
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)

    return res

# Example usage
# context = """
# Pakistani-Canadian columnist Tarek Fatah passed away on 24 April after a prolonged battle with Cancer. Born in Karachi, Pakistan before emigrating to Canada in 1987, Fatah was an award-winning reporter, columnist, and radio and television commentator, both in Canada and abroad. Fatah, who died at 73, was a political activist, a fierce defender of human rights and a staunch opponent of religious fanaticism in any form, nothing scared Tarek Fatah. He also authored several books including, 'Chasing a Mirage: The Tragic Illusion of an Islamic State' and 'The Jew is Not My Enemy: Unveiling the Myths that Fuel Muslim Anti-Smitism.' Mr. Fatah was known for his progressive views on Islam and his fiery stance on Pakistan. He called himself an 'Indian born in Pakistan' and a 'Punjabi born into Islam'. He won awards from organizations such as the Donor Prize, Helen and the Stan Wine Canadian Book Award, and was known for frequent commentary in Canadian, Indian, and international media.
# """

# question = 'Tarek Fatah passed away on?'

# result = four(context, question)
# print(result)


# %%
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Function to summarize context chunks
def summarize_context(context, max_chunk_length=500, max_summary_length=1000, top_chunks=5):
    # Split the context into chunks of maximum length
    context_chunks = [context[i:i+max_chunk_length] for i in range(0, len(context), max_chunk_length)]
    
    # Initialize an empty list to store the summaries
    summaries = []
    
    # Summarize each chunk separately
    for chunk in context_chunks[:top_chunks]:
        # Determine the appropriate max_length based on the length of the input chunk
        max_length = min(len(chunk) * 2, max_summary_length)
        # Summarize the chunk with the dynamically determined max_length
        summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return summaries




# %%
import tkinter as tk
from tkinter import ttk
from bs4 import BeautifulSoup
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(news_title):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(news_title)
    compound_score = sentiment_score['compound']
    if compound_score > 0:
        return 'Positive'
    elif compound_score < 0:
        return 'Negative'
    else:
        return 'Neutral'
def final(user_input):
    print(user_input)
    one()
    tokens, line_numbers ,user_input= perform_all(user_input)
    context = three(line_numbers)
    result = four(context, user_input)
    print(result)
    
    
    st=''
    ARTICLE = context
    summarized_chunks = summarize_context(ARTICLE)
    for index, chunk_summary in enumerate(summarized_chunks, 1):
        st=st+f"{index}. {chunk_summary}"
        st=st+"\n"
        print(f"{index}. {chunk_summary}")
    
    return str(result['answer']),st
def scrape_news(year_select, month_select, day_select, text_language):
    if text_language == "English":
        text_language = "1"
    else:
        text_language="2"
    url = (
        "https://sarkaripariksha.com/gk-and-current-affairs/"
        + year_select
        + "/"
        + month_select
        + "/"
        + str(day_select)
        + "/"
        + text_language
        + "/"
    )
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    news_data = []
    news_list = soup.find_all("div", class_="examlist-details-img-box")
    
    for news_item in news_list:
        news_title = news_item.find("h2").find("a").get_text(strip=True)
        href_link = news_item.find("h2").find("a")["href"]
        
        # Function to extract the category from the HTML content of a news article
        category_set = extract_category(href_link)
        
        # Analyze sentiment
        sentiment = analyze_sentiment(news_title)
        
        news_data.append({"Title": news_title, "Category": category_set, "Link": href_link, "Sentiment": sentiment})
    
    return pd.DataFrame(news_data)

def extract_category(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    category_div = soup.find("div", class_="CategoryCurrentAffairsBox")
    return category_div.find("span").text.replace('Category :', '').strip()

def load_and_display():
    year_select = year_var.get()
    month_select = month_var.get()
    day_select = day_var.get()
    text_language = language_var.get()
    
    try:
        news_df = scrape_news(year_select, month_select, day_select, text_language)
        news_df.to_csv("news_data.csv", index=False)
        category_select = category_var.get()
        
        news_df = pd.read_csv("news_data.csv")

        if news_df.empty:
            output_text.config(state=tk.NORMAL)
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, "Data not available for that date.")
            output_text.config(state=tk.DISABLED)
            return
        
        if category_select != "All":
            news_df = news_df[news_df['Category'] == category_select]

        context = ""
        for index, row in news_df.iterrows():
            context += f"News {index+1} : {row['Title']}\nUrl: {row['Link']}\n"


        output_text.config(state=tk.NORMAL)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, context)
        output_text.config(state=tk.DISABLED)

    except AttributeError as e:
        output_text.config(state=tk.NORMAL)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "Data not available for this date")
        output_text.config(state=tk.DISABLED)

def process_input():
    # Retrieve input from the input text box
    user_input = input_textbox.get("1.0", "end-1c")
    
    # Process input (here we just reverse the text for demonstration)
    processed_output, p = final(str(user_input))
    
    # Clear the output textbox and then update it with the processed output
    output_textbox.delete("1.0", "end")
    output_textbox.insert("1.0", processed_output)

    output_textbox_1.delete("1.0", "end")
    output_textbox_1.insert("1.0", p)

# Set up the main window
root = tk.Tk()
root.title("NEWS GENERATOR")

# Set up the frame for layout
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

# Create the input text box
input_label = tk.Label(frame, text="Enter your text:")
input_label.pack()
input_textbox = tk.Text(frame, height=2, width=40)
input_textbox.pack(padx=5, pady=5)

# Create the button to trigger processing
process_button = tk.Button(frame, text="Process", command=process_input)
process_button.pack(pady=10)

# Create the output text box
output_label = tk.Label(frame, text="Processed Answer:")
output_label.pack()
output_textbox = tk.Text(frame, height=2, width=40)
output_textbox.pack(padx=5, pady=5)

output_label_1 = tk.Label(frame, text="Processed News:")
output_label_1.pack()
output_textbox_1 = tk.Text(frame, height=14, width=100)
output_textbox_1.pack(padx=5, pady=5)

# Frame for date selection
date_frame = ttk.Frame(root)
date_frame.pack(pady=10)

year_var = tk.StringVar()
year_label = ttk.Label(date_frame, text="Select year:")
year_label.grid(row=0, column=0, padx=5, pady=5)
year_select = ttk.Combobox(date_frame, textvariable=year_var, values=["2024", "2023", "2022"])
year_select.grid(row=0, column=1, padx=5, pady=5)
year_select.current(0)

month_var = tk.StringVar()
month_label = ttk.Label(date_frame, text="Select month:")
month_label.grid(row=0, column=2, padx=5, pady=5)
month_select = ttk.Combobox(date_frame, textvariable=month_var, values=[
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
])
month_select.grid(row=0, column=3, padx=5, pady=5)
month_select.current(0)

day_var = tk.StringVar()
day_label = ttk.Label(date_frame, text="Select day:")
day_label.grid(row=0, column=4, padx=5, pady=5)
day_select = ttk.Combobox(date_frame, textvariable=day_var, values=[str(i) for i in range(1, 32)])
day_select.grid(row=0, column=5, padx=5, pady=5)
day_select.current(0)

# Frame for language selection
language_frame = ttk.Frame(root)
language_frame.pack(pady=10)

language_var = tk.StringVar()
language_label = ttk.Label(language_frame, text="Select Language:")
language_label.grid(row=0, column=0, padx=5, pady=5)
language_select = ttk.Combobox(language_frame, textvariable=language_var, values=["English", "Hindi"])
language_select.grid(row=0, column=1, padx=5, pady=5)
language_select.current(0)

# Frame for category selection
category_frame = ttk.Frame(root)
category_frame.pack(pady=10)

category_var = tk.StringVar()
category_label = ttk.Label(category_frame, text="Select Category:")
category_label.grid(row=0, column=0, padx=5, pady=5)
category_select = ttk.Combobox(category_frame, textvariable=category_var, values=[
    "All", "Business and economics", "Sports", "National", "International",
    "Defense", "State", "Appointment/Resignation", "Awards",
    "Science and Tech", "Miscellaneous"
])
category_select.grid(row=0, column=1, padx=5, pady=5)
category_select.current(0)

# Button to load and display news
load_button = ttk.Button(root, text="Load and Display News", command=load_and_display)
load_button.pack(pady=10)

# Output text widget to display news
output_text = tk.Text(root, height=20, width=100)
output_text.pack(pady=10)
output_text.config(state=tk.DISABLED)

root.mainloop()


# %%



