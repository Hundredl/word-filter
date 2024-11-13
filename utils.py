import nltk
from collections import Counter
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

# Ensure the Brown corpus is downloaded (only needed on first run)
from nltk.corpus import brown

def get_brown_word_freq():
    nltk.download('brown')

    # Load Brown corpus word frequencies
    brown_words = brown.words()
    brown_word_freq = Counter(brown_words)
    return brown_word_freq

def frequency_level(word, brown_word_freq):
    """
    Classify the frequency of a word based on its count in the Brown corpus.
    
    Args:
        word (str): The word to classify.
    
    Returns:
        str: "High frequency", "Medium frequency", or "Low frequency" based on the frequency threshold.
    """
    freq = brown_word_freq.get(word, 0)
    if freq >= 500:  # Threshold for high frequency
        return "High frequency"
    elif 100 <= freq < 500:  # Threshold for medium frequency
        return "Medium frequency"
    else:  # Low frequency
        return "Low frequency"

def meaning(word):
    """
    Retrieve the Chinese meaning of a word using Youdao Dictionary.
    
    Args:
        word (str): The word for which to retrieve the meaning.
    
    Returns:
        str: The Chinese meaning of the word.
    """
    url = f'https://dict.youdao.com/result?word={word}&lang=en'
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Cookie': 'OUTFOX_SEARCH_USER_ID=-1090623694@124.89.8.164'
    }
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    result = soup.select('.simple.dict-module .word-exp')
    return ''.join([e.getText() for e in result])

def generate_frequency_table(text, add_chinese_meaning = False):
    """
    Generate a frequency table for words in a given text, including data from the Brown corpus,
    frequency level, occurrence in the text, and example sentences.

    Args:
        text (str): The input text for which to generate the frequency table.
    
    Returns:
        pd.DataFrame: A DataFrame containing word, Brown corpus frequency, frequency level,
                      frequency in the text, example sentence, and Chinese meaning.
    """
    brown_word_freq = get_brown_word_freq()
    # Calculate word frequencies in the input text
    words = re.findall(r'\b\w+\b', text.lower())
    article_word_freq = Counter(words)

    # Split text into sentences for example extraction
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [sentence.replace("\n", " ") for sentence in sentences]

    # Construct frequency table data
    table_data = []
    for word, freq in article_word_freq.items():
        brown_freq = brown_word_freq.get(word, 0)
        level = frequency_level(word, brown_word_freq)

        # Find the first sentence containing the word as an example sentence
        example_sentence = next((sentence for sentence in sentences if word in sentence.lower()), "")
        # Highlight the word in the example sentence
        example_sentence = re.sub(r'\b' + word + r'\b', f'【{word}】', example_sentence)

        # Append data for the frequency table
        table_data.append([word, brown_freq, level, freq, example_sentence])
    
    # Create a DataFrame for the frequency table
    df = pd.DataFrame(table_data, columns=["Word", "Brown Corpus Frequency", "Frequency Level", "Text Frequency", "Example Sentence"])
    # Filter and reorder columns
    df = df[["Brown Corpus Frequency", "Text Frequency", "Word", "Example Sentence"]]
    # # Add a new column "Check" with all values set to 0
    # df["Check"] = False
    # # Reorder columns to place "Check" before "Word"
    # df = df[["Brown Corpus Frequency", "Text Frequency", "Check", "Word", "Example Sentence"]]
    # Remove purely numeric words
    df = df[~df['Word'].str.isnumeric()]
    # Exclude words with high frequency (>50) in the Brown corpus
    df = df[df['Brown Corpus Frequency'] <= 50]
    # Add Chinese meaning for specific words
    if add_chinese_meaning:
        df['Chinese Meaning'] = df.apply(lambda x: meaning(x['Word']) if x['Brown Corpus Frequency'] <= 50 else '', axis=1)
    # Sort by text frequency in descending order
    df = df.sort_values(by="Brown Corpus Frequency", ascending=False)
    # reindex
    df = df.reset_index(drop=True)
    df.index += 1


    return df


