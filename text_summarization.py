def _create_frequency_matrix(sentences):
    freq_matrix = {}
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sentence in sentences:
        freq_table = {}
        words = word_tokenize(sentence)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stop_words:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        freq_matrix[sentence] = freq_table

    return freq_matrix

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sentence, f_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence
        tf_matrix[sentence] = tf_table

    return tf_matrix

def _create_documents_per_words(freq_matrix):
    word_document_count = {}

    for sentence, f_table in freq_matrix.items():
        for word in f_table.keys():
            if word in word_document_count:
                word_document_count[word] += 1
            else:
                word_document_count[word] = 1

    return word_document_count

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sentence, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
        idf_matrix[sentence] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sentence1, f_table1), (sentence2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word, tf_value), (_, idf_value) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word] = tf_value * idf_value
        tf_idf_matrix[sentence1] = tf_idf_table

    return tf_idf_matrix

def _score_sentences(tf_idf_matrix):
    sentence_scores = {}

    for sentence, f_table in tf_idf_matrix.items():
        total_score_per_sentence = sum(f_table.values())
        sentence_scores[sentence] = total_score_per_sentence

    return sentence_scores

def _find_average_score(sentence_scores):
    sum_values = sum(sentence_scores.values())
    average_score = sum_values / len(sentence_scores)

    return average_score

def _generate_summary(sentences, sentence_scores, threshold):
    summary = []

    for sentence in sentences:
        if sentence in sentence_scores and sentence_scores[sentence] >= threshold:
            summary.append(sentence)

    return " ".join(summary)

import math
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords    

# Define all functions here (refer to the definitions provided above)

# Example text input
text = """In the distant land of Aleron, nestled between towering mountains and verdant forests, there lived a young woman named Ilara. She was not of noble birth, nor did she possess great wealth or power. Yet, her heart was filled with a curiosity and wonder that had been her constant companion since childhood. While the people of her village were content with their simple lives, Ilara yearned for adventure beyond the boundaries of what she knew. One autumn evening, as Ilara sat by the village’s great oak tree, a peculiar sight caught her eye. A glowing figure appeared at the edge of the forest. It was an old man dressed in robes that shimmered with an ethereal light, carrying a staff adorned with crystals that glowed with a strange warmth. Alithor spoke of a prophecy: a young woman from the village would play a crucial role in restoring balance to the land. Dark forces had begun to stir again, spreading fear and chaos across Aleron. Ancient powers, once dormant, were awakening. Evil sorcerers sought to control these powers, threatening to unleash destruction upon the world. Alithor believed that Ilara was the chosen one who could restore harmony to the land. Without hesitation, Ilara accepted her calling. Alithor took her under his wing, teaching her the ancient arts of magic. She learned to wield fire and water, to call upon the wind, and to sense the heartbeat of the earth itself. As her training progressed, Ilara grew stronger, her powers deepening and becoming a part of her very soul. But Alithor warned her: power alone would not be enough. She would need wisdom, courage, and resilience, for the path ahead was fraught with peril. One morning, as the first light of dawn crept over the mountains, Alithor gifted Ilara a map. It led to the Valley of Shadows, where a powerful relic known as the Crystal of Eternity lay hidden. This crystal held the power to seal the forces of darkness forever. However, the journey to the valley was treacherous, and Ilara would encounter trials that would test every ounce of her strength and resolve. Setting out on her journey, Ilara traversed rugged terrain and dense forests, crossed rivers and braved storms. Along the way, she encountered allies who joined her cause: Aiden, a skilled swordsman haunted by his own past, and Lyra, a healer with a mysterious connection to the spirit world. Together, they formed a bond that grew stronger with each challenge they faced. The battle was fierce, with Ilara and her friends fighting valiantly against creatures that seemed to melt into the shadows. Just when hope seemed lost, Ilara summoned a powerful spell, unleashing a wave of light that scattered the creatures and saved her friends. But the fight took a toll on her, and she realized that Malakar was a foe far more dangerous than she had imagined. As they neared the Valley of Shadows, Ilara and her companions encountered a group of villagers who had been enslaved by Malakar’s minions. Determined to help, Ilara used her powers to free them, rallying the villagers to resist the forces of darkness. Her act of bravery inspired others, and soon, whispers of her deeds spread across Aleron. People began to hope again, and a resistance was born. Finally, after weeks of hardship, Ilara reached the Valley of Shadows. She stood before the Temple of Eternity, an ancient structure that had stood since time immemorial. But Malakar was waiting for her. In a climactic battle, Ilara faced the dark sorcerer, wielding every skill and ounce of courage she possessed. The earth shook, and the skies darkened as their powers clashed. Just when it seemed Malakar would overpower her, Ilara remembered Alithor’s teachings. Calling upon the wisdom and strength within, she channeled the power of the Crystal of Eternity. With a blinding light, Malakar was defeated, his darkness dissolved into the air like mist at dawn. The crystal, now pulsating with a soft, serene glow, had been returned to its rightful place. Peace was restored to Aleron, and Ilara, though weary, felt a sense of fulfillment and purpose. She had become more than she ever dreamed, not just a wielder of magic, but a beacon of hope and strength for all. Upon returning to her village, she found that she was celebrated as a hero. But more than the praise, what she cherished most was the knowledge that she had protected her world and inspired others to stand for what was right."""

# 1 Sentence Tokenize
sentences = sent_tokenize(text)
total_documents = len(sentences)
#print(sentences)

# 2 Create the Frequency matrix of the words in each sentence.
freq_matrix = _create_frequency_matrix(sentences)
#print(freq_matrix)

# 3 Calculate TermFrequency and generate a matrix
tf_matrix = _create_tf_matrix(freq_matrix)
#print(tf_matrix)

# 4 creating table for documents per words
count_doc_per_words = _create_documents_per_words(freq_matrix)
#print(count_doc_per_words)

# 5 Calculate IDF and generate a matrix
idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
#print(idf_matrix)

# 6 Calculate TF-IDF and generate a matrix
tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
#print(tf_idf_matrix)

# 7 Important Algorithm: score the sentences
sentence_scores = _score_sentences(tf_idf_matrix)
#print(sentence_scores)

# 8 Find the threshold
threshold = _find_average_score(sentence_scores)
print(threshold)

# 9 Important Algorithm: Generate the summary
summary = _generate_summary(sentences, sentence_scores, threshold)
print(summary)
