# CSE508_Winter2024_Final_Project
1. Introduction:
The project introduces a novel news aggregation platform prioritizing user privacy and personalization. It utilizes dynamic web crawling, advanced NLP, and machine learning to deliver tailored news experiences without storing identifiable data. The platform offers real-time summarization, adapts to user preferences, and provides instant explanations for terms

# News Generator Application

This repository contains Python code for a News Generator application. The application consists of several functionalities, including:

1. **Inverted Index Creation and Querying**: Creates an inverted index from a CSV file containing formatted news data. Allows users to query the inverted index based on user input and retrieve relevant line numbers.

2. **Question Answering with Transformer Models**: Utilizes Transformer-based models for question-answering tasks. Given a context and a question, the application extracts the answer using a pre-trained model.

3. **News Scraping and Summarization**: Scrapes news articles from a website, extracts relevant information, and provides a summary of the content.

4. **Graphical User Interface (GUI)**: Provides a user-friendly interface for interacting with the application, including options for selecting news categories and languages, inputting text for processing, and displaying results.


## Usage

### Setup

1. Install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

### Cloning the Repository

To clone this repository and set up the News Generator application locally, run the following command in your terminal:

```bash
git clone https://github.com/your-username/news-generator.git
cd news-generator
```

### Running the Application

1. Run the `news_generator.py` script to launch the GUI application.

```bash
python news_generator.py
```

2. Use the GUI interface to input text or select news categories and languages.
3. Click on the "Process" button to execute the selected functionality.
4. View the processed output displayed in the application interface.


## Dependencies

- Python 3.x
- Libraries: `transformers`, `nltk`, `pandas`, `beautifulsoup4`, `requests`, `vaderSentiment`, `tkinter`

## Contributors

- [Contributor Name](https://github.com/contributor): Description of contributions.
