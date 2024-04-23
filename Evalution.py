# %%
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")

# Read context from the text file
with open('data.txt', "r") as file:
    context = file.read().replace('\n', ' ')

# Split context into lines using full stops
lines = context.split('.')

# Initialize a list to store generated questions and answers
generated_data = []

# Generate questions and answers for each line
for line in lines:
    inputs = tokenizer("generate question:", line.strip(), return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")

    # Split question and answer if [SEP] token exists, else consider the entire output as the question
    if tokenizer.sep_token in question_answer:
        question, answer = question_answer.split(tokenizer.sep_token)
        if answer.strip():
            generated_data.append([context.strip(), question.strip(), answer.strip()])

# Save generated data to a CSV file
with open("generated_data.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Context", "Question", "Answer"])
    writer.writerows(generated_data)

print("Generated data saved to generated_data.csv")


# %%
import csv
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def generate_answers(input_file, output_file, encoding='utf-8', delimiter=','):
    # Load model and tokenizer
    model_name = "deepset/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a question-answering pipeline
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    # Open input CSV file and output CSV file for writing
    with open(input_file, 'r', newline='', encoding=encoding) as csvfile_in, \
         open(output_file, 'w', newline='', encoding=encoding) as csvfile_out:

        reader = csv.DictReader(csvfile_in, delimiter=delimiter)
        fieldnames = ['Context', 'Question', 'Answer']
        writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()

        # Process each row in the input CSV file
        for row in reader:
            context = row['Context']
            question = row['Question']

            # Get answer using the question-answering pipeline
            QA_input = {
                'question': question,
                'context': context
            }
            result = nlp(QA_input)

            # Write context, question, and answer to the output CSV file
            writer.writerow({
                'Context': context,
                'Question': question,
                'Answer': result['answer']
            })

# Specify input and output CSV file paths
input_csv_file = 'generated_data.csv'
output_csv_file = 'generated_answers.csv'

# Specify the encoding and delimiter used in your input CSV file
csv_encoding = 'latin1'  # Example: 'latin1' or 'ISO-8859-1'
csv_delimiter = ','  # Example: ',' or ';' depending on the delimiter used in your CSV

# Generate answers for each context and question pair in the input CSV file
generate_answers(input_csv_file, output_csv_file, encoding=csv_encoding, delimiter=csv_delimiter)


# %%
# # Install sentence-transformers library
# !pip install sentence-transformers


# %%
import csv
from sentence_transformers import SentenceTransformer, util

def compute_relevance_scores(labeled_dataset_file, generated_answers_file, output_file):
    # Load the SentenceTransformer model for semantic similarity
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Load labeled dataset and generated answers
    labeled_data = load_csv_data(labeled_dataset_file)
    generated_data = load_csv_data(generated_answers_file)

    # Create output CSV file for storing results
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile_out:
        fieldnames = ['Context', 'Question', 'Generated_Answer', 'Labeled_Answer', 'Relevance_Score']
        writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each entry in the generated answers
        for generated_entry in generated_data:
            generated_context = generated_entry['Context']
            generated_question = generated_entry['Question']
            generated_answer = generated_entry['Answer']

            # Find the corresponding labeled answer based on context and question
            labeled_answer = find_labeled_answer(labeled_data, generated_context, generated_question)

            if labeled_answer:
                # Compute similarity score between generated answer and labeled answer
                relevance_score = compute_similarity_score(model, generated_answer, labeled_answer)

                # Write the results to the output CSV file
                writer.writerow({
                    'Context': generated_context,
                    'Question': generated_question,
                    'Generated_Answer': generated_answer,
                    'Labeled_Answer': labeled_answer,
                    'Relevance_Score': relevance_score
                })

def load_csv_data(csv_file):
    # Load data from CSV file into a list of dictionaries
    data = []
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def find_labeled_answer(labeled_data, context, question):
    # Find the labeled answer corresponding to the given context and question
    for entry in labeled_data:
        if entry['Context'] == context and entry['Question'] == question:
            return entry['Answer']
    return None

def compute_similarity_score(model, text1, text2):
    # Compute similarity score between two text strings using SentenceTransformer model
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()

# Specify file paths for labeled dataset and generated answers
labeled_dataset_file = 'generated_data.csv'
generated_answers_file = 'generated_answers.csv'
output_file = 'relevance_scores.csv'

# Compute relevance scores between generated answers and labeled answers
compute_relevance_scores(labeled_dataset_file, generated_answers_file, output_file)


# %%
import csv
from sentence_transformers import SentenceTransformer, util
import numpy as np

def compute_relevance_scores(labeled_dataset_file, generated_answers_file, output_file):
    # Load the SentenceTransformer model for semantic similarity
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Load labeled dataset and generated answers
    labeled_data = load_csv_data(labeled_dataset_file)
    generated_data = load_csv_data(generated_answers_file)

    # Create output CSV file for storing results
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile_out:
        fieldnames = ['Context', 'Question', 'Generated_Answer', 'Labeled_Answer', 'Relevance_Score']
        writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each entry in the generated answers
        for generated_entry in generated_data:
            generated_context = generated_entry['Context']
            generated_question = generated_entry['Question']
            generated_answer = generated_entry['Answer']

            # Find the corresponding labeled answer based on context and question
            labeled_answer = find_labeled_answer(labeled_data, generated_context, generated_question)

            if labeled_answer:
                # Compute similarity score between generated answer and labeled answer
                relevance_score = compute_similarity_score(model, generated_answer, labeled_answer)

                # Write the results to the output CSV file
                writer.writerow({
                    'Context': generated_context,
                    'Question': generated_question,
                    'Generated_Answer': generated_answer,
                    'Labeled_Answer': labeled_answer,
                    'Relevance_Score': relevance_score
                })

    # Calculate overall relevance score for each context
    calculate_overall_relevance(output_file)

def load_csv_data(csv_file):
    # Load data from CSV file into a list of dictionaries
    data = []
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def find_labeled_answer(labeled_data, context, question):
    # Find the labeled answer corresponding to the given context and question
    for entry in labeled_data:
        if entry['Context'] == context and entry['Question'] == question:
            return entry['Answer']
    return None

def compute_similarity_score(model, text1, text2):
    # Compute similarity score between two text strings using SentenceTransformer model
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()

def calculate_overall_relevance(output_file):
    # Load relevance scores from the output CSV file
    relevance_scores = []
    with open(output_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            relevance_scores.append(float(row['Relevance_Score']))

    # Compute overall relevance score (average of all relevance scores)
    overall_relevance_score = np.mean(relevance_scores)
    print(f"Overall Relevance Score: {overall_relevance_score}")

# Specify file paths for labeled dataset and generated answers
labeled_dataset_file = 'generated_data.csv'
generated_answers_file = 'generated_answers.csv'
output_file = 'relevance_scores_1.csv'

# Compute relevance scores between generated answers and labeled answers
compute_relevance_scores(labeled_dataset_file, generated_answers_file, output_file)


# %%
import csv
from rouge_score import rouge_scorer
import numpy as np

def compute_relevance_scores(labeled_dataset_file, generated_answers_file):
    # Load labeled dataset and generated answers
    labeled_data = load_csv_data(labeled_dataset_file)
    generated_data = load_csv_data(generated_answers_file)

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize lists to store ROUGE scores for all entries
    rouge1_precisions, rouge1_recalls, rouge1_f1s = [], [], []
    rouge2_precisions, rouge2_recalls, rouge2_f1s = [], [], []
    rougeL_precisions, rougeL_recalls, rougeL_f1s = [], [], []

    # Iterate over each entry in the generated answers
    for generated_entry in generated_data:
        generated_context = generated_entry['Context']
        generated_question = generated_entry['Question']
        generated_answer = generated_entry['Answer']

        # Find the corresponding labeled answer based on context and question
        labeled_answer = find_labeled_answer(labeled_data, generated_context, generated_question)

        if labeled_answer:
            # Calculate ROUGE scores between generated answer and labeled answer
            rouge_scores = scorer.score(generated_answer, labeled_answer)

            # Extract ROUGE-1 scores
            rouge1_precisions.append(rouge_scores['rouge1'].precision)
            rouge1_recalls.append(rouge_scores['rouge1'].recall)
            rouge1_f1s.append(rouge_scores['rouge1'].fmeasure)

            # Extract ROUGE-2 scores
            rouge2_precisions.append(rouge_scores['rouge2'].precision)
            rouge2_recalls.append(rouge_scores['rouge2'].recall)
            rouge2_f1s.append(rouge_scores['rouge2'].fmeasure)

            # Extract ROUGE-L scores
            rougeL_precisions.append(rouge_scores['rougeL'].precision)
            rougeL_recalls.append(rouge_scores['rougeL'].recall)
            rougeL_f1s.append(rouge_scores['rougeL'].fmeasure)

    # Calculate average ROUGE scores across all entries
    avg_rouge1_precision = np.mean(rouge1_precisions)
    avg_rouge1_recall = np.mean(rouge1_recalls)
    avg_rouge1_f1 = np.mean(rouge1_f1s)

    avg_rouge2_precision = np.mean(rouge2_precisions)
    avg_rouge2_recall = np.mean(rouge2_recalls)
    avg_rouge2_f1 = np.mean(rouge2_f1s)

    avg_rougeL_precision = np.mean(rougeL_precisions)
    avg_rougeL_recall = np.mean(rougeL_recalls)
    avg_rougeL_f1 = np.mean(rougeL_f1s)

    # Print overall ROUGE scores
    print("Overall ROUGE Scores:")
    print(f"ROUGE-1 Precision: {avg_rouge1_precision}, Recall: {avg_rouge1_recall}, F1 Score: {avg_rouge1_f1}")
    print(f"ROUGE-2 Precision: {avg_rouge2_precision}, Recall: {avg_rouge2_recall}, F1 Score: {avg_rouge2_f1}")
    print(f"ROUGE-L Precision: {avg_rougeL_precision}, Recall: {avg_rougeL_recall}, F1 Score: {avg_rougeL_f1}")

def load_csv_data(csv_file):
    # Load data from CSV file into a list of dictionaries
    data = []
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def find_labeled_answer(labeled_data, context, question):
    # Find the labeled answer corresponding to the given context and question
    for entry in labeled_data:
        if entry['Context'] == context and entry['Question'] == question:
            return entry['Answer']
    return None

# Specify file paths for labeled dataset and generated answers
labeled_dataset_file = 'generated_data.csv'
generated_answers_file = 'generated_answers.csv'

# Compute relevance scores including overall ROUGE scores
compute_relevance_scores(labeled_dataset_file, generated_answers_file)



