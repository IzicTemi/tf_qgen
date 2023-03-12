from requests import session
import streamlit as st 
import requests
import json
import string
import re
import random
import spacy
import torch
import scipy
from spacy.tokens import Doc
from summarizer import Summarizer
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
from negate import Negator

st.title('True False Question Generator')

nlp = spacy.load('en_core_web_md')
# nlp.add_pipe('merge_noun_chunks')
nlp.add_pipe('merge_entities')

negator = Negator()

def load_references_model():
    nlp_coref = spacy.load("en_coreference_web_trf")

    # use replace_listeners for the coref components
    nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
    nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

    # we won't copy over the span cleaner
    nlp.add_pipe("coref", source=nlp_coref)
    nlp.add_pipe("span_resolver", source=nlp_coref)


# st.set_page_config(page_title='True False Question Generator', page_icon=":shark:", layout="wide")
@st.cache_data(experimental_allow_widgets=True)
def get_data():
    context = st.text_area("Context", height=300)
    return context

context = get_data()

resolver = st.checkbox("Use coreference resolution. Only select if RAM > 8gb", value = False)

generate = st.button("Generate!")

# @st.cache_resource
def resolve_references(doc: Doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    load_references_model()
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
            
            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string

@st.cache_resource
def init_summarizer_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    custom_config = AutoConfig.from_pretrained('distilbert-base-uncased')
    custom_config.output_hidden_states=True
    custom_model = AutoModel.from_pretrained('distilbert-base-uncased', config=custom_config)

    model = Summarizer(custom_model=custom_model, custom_tokenizer=tokenizer)
    return model, custom_config, tokenizer

model, custom_config, tokenizer= init_summarizer_model()

@st.cache_resource
def summarize(text):
    res = model.calculate_optimal_k(text, k_max=10)
    result = model(text, min_length=40, max_length=250, num_sentences=5)
    summarized = ''.join(result)

    summarized_doc = nlp(summarized)
    sentence_list = [sent.text for sent in summarized_doc.sents]

    return sentence_list


@st.cache_resource
def init_generation():
    mask_model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased', config=custom_config)
    sent_model = SentenceTransformer('distilbert-base-uncased')

    return (mask_model, sent_model)

@st.cache_resource
def generate_mask(text):
    doc = nlp(text)
    masked_sentences = []
    for num in range(len(list(doc.noun_chunks))):
        test=''.join(['[MASK]' + token.whitespace_ if token.text in [str(a) for a in list(doc.noun_chunks)][num] and not token.is_stop else token.text + token.whitespace_ for token in doc])
        if '[MASK]' in test:
            masked_sentences.append(test)
    return masked_sentences

def generate_similar_sentences(mask_model, masked_sentence):
    output_list = []
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        logits = mask_model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    predicted_token_id = torch.flip(logits[0, mask_token_index].argsort(axis=-1), [0, 1])[:, :5]
    
    outputs = []
    for token in predicted_token_id:
        output = tokenizer.convert_ids_to_tokens(token)
        outputs.append(output)
        
    for a in zip(*outputs[::-1]):
        sentence = masked_sentence
        for word in a:
            sentence = sentence.replace("[MASK]", word, 1)
            output_list.append(sentence)

    return output_list

def sort_by_similarity(sent_model, original_sentence, generated_sentences_list):
    sentence_embeddings = sent_model.encode(generated_sentences_list)

    queries = [original_sentence]
    query_embeddings = sent_model.encode(queries)
    
    # Find the top sentences of the corpus for each query sentence based on cosine similarity
    dissimilar_sentences = []

    for _, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])


        for idx, distance in results[0:3]:
            dissimilar_sentences.append(generated_sentences_list[idx].strip())
          
    return dissimilar_sentences

def create_tf_qn(sentence, method):
    mask_model, sent_model = init_generation()

    if method == 'negation':
        false_sentence = negator.negate_sentence(sentence)
    elif method == 'masking':
        masked_sentences = generate_mask(sentence)
        # print(masked_sentences)
        masked_sentence = random.choice(masked_sentences)
        generated_sentences_list = generate_similar_sentences(mask_model, masked_sentence)
        false_sentences = sort_by_similarity(sent_model, sentence, generated_sentences_list)
        false_sentence = false_sentences[0]
    return false_sentence


if generate:

    with st.spinner(text="In progress"):


        if resolver:
            text = [context]
            resolved_txt = [resolve_references(coref_doc) for coref_doc in nlp.pipe(text)]

            sentence_list = summarize(resolved_txt[0])
        else:
             sentence_list = summarize(context)

        sentence = random.choice(sentence_list)
        show = random.choices([True, False], cum_weights=(20, 80), k=1)
        if show[0] is True:
            final_sentence = sentence
        else:
            method = random.choices(['negation', 'masking'], cum_weights=(20, 80), k=1)
            final_sentence = create_tf_qn(sentence, method[0])
        st.info(final_sentence)

        def check_true():
            if True == show[0]:
                st.success('Correct')
            else:
                st.error('Wrong')

        def check_false():
            if False == show[0]:
                st.success('Correct')
            else:
                st.error('Wrong')

        if st.button('True', on_click=check_true):
            check_true()
        elif st.button('False', on_click=check_false):
            check_false()

           