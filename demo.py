# from transformers import pipeline

# classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = ["I am not having a great day"]

# model_outputs = classifier(sentences)
# print(model_outputs[0])
# # produces a list of dicts for each of the labels


from tokenizers import Tokenizer
import onnxruntime as ort

from os import cpu_count
import numpy as np  # only used for the postprocessing sigmoid
import re
import nltk
import pandas as pd
# nltk.download('stopwords')
# nltk.download('punkt')

labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

def preprocess(sentences):
    preprocessed_sentences = []
    stop_words = nltk.corpus.stopwords.words("english")
   
    
    for sentence in sentences:
        # Lowercase the sentence
        sentence = sentence.lower()
        
        # Remove URLs
        sentence = re.sub(r'http\S+', '', sentence)
        
        # Remove special characters and punctuation
        sentence = re.sub(r'[^A-Za-z0-9\s]+', '', sentence)
        
        # Tokenize the sentence
        words = nltk.word_tokenize(sentence)
        
        filtered_sentence = [word for word in words]
        
        # Join the words back into a sentence
        sentence = ' '.join(filtered_sentence)
        
        # Remove extra whitespaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        preprocessed_sentences.append(sentence)
    
    return preprocessed_sentences

def convert_to_csv(emotion_data):
    output_file = "data.csv"

    df = pd.DataFrame(emotion_data.items(), columns=['Emotion', 'Count'])
    df.to_csv(output_file, index=False)


def inference(sentences):
    # labels as (ordered) list - from the go_emotions dataset
    
    tokenizer = Tokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

    # Optional - set pad to only pad to longest in batch, not a fixed length.
    # (without this, the model will run slower, esp for shorter input strings)
    params = {**tokenizer.padding, "length": None}
   
    tokenizer.enable_padding(**params)

    # print(sentences)
    sentences = preprocess(sentences)
    # print(sentences)

    tokens_obj = tokenizer.encode_batch(sentences)
    # print(tokens_obj)
    
    # exit()

    def load_onnx_model(model_filepath):
        _options = ort.SessionOptions()
        _options.inter_op_num_threads, _options.intra_op_num_threads = cpu_count(), cpu_count()
        _providers = ["CPUExecutionProvider"]  # could use ort.get_available_providers()
        return ort.InferenceSession(path_or_bytes=model_filepath, sess_options=_options, providers=_providers)

    model = load_onnx_model("model_quantized.onnx")

    output_names = [model.get_outputs()[0].name]  # E.g. ["logits"]

    input_feed_dict = {
    "input_ids": [t.ids for t in tokens_obj],
    "attention_mask": [t.attention_mask for t in tokens_obj]
    }

    logits = model.run(output_names=output_names, input_feed=input_feed_dict)
    print(logits)
    # produces a numpy array, one row per input item, one col per label

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Post-processing. Gets the scores per label in range.
    # Auto done by Transformers' pipeline, but we must do it manually with ORT.
    model_outputs = sigmoid(logits[0]) 

    # for example, just to show the top result per input item

    print(model_outputs)

    emotion_data = {}
    # for label in labels:
    #     emotion_data[label] = 0 
     
    for probas in model_outputs:
        top_result_index = np.argmax(probas)
        # print(top_result_index,len(probas))
        # print(labels[top_result_index], "with score:", probas[top_result_index])
        label = labels[top_result_index]
        value = probas[top_result_index]*100
        emotion_data[label] = 1 + emotion_data.get(label,0)
    
    convert_to_csv(emotion_data)
    
inference(sentences)
