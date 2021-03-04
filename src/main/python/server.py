from flask import Flask, request, jsonify
app = Flask(__name__)

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans

barthez_tokenizer = AutoTokenizer.from_pretrained("moussaKam/mbarthez")
barthez = AutoModel.from_pretrained("moussaKam/mbarthez", output_hidden_states=True)
barthez.eval()

def get_vectors(utterance):
    encoded_input = barthez_tokenizer(utterance, return_tensors='pt')
    hidden_states = barthez(**encoded_input)[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    word_instances = []
    tokens = ("<s> " + utterance + " </s>").split()
    token_pointer = 0
    found = False
    concat_token = ""
    vectors = []
    tmp_vectors = []
    for i, token_str in enumerate(barthez_tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])):
        concat_token += token_str.replace("▁", "")
        tmp_vectors.append(token_embeddings[-1][i].detach().numpy())
        if concat_token == tokens[token_pointer]:
            vectors.append(np.mean(tmp_vectors, axis=0))
            found = False
            concat_token = ""
            token_pointer += 1
            tmp_vectors = []
    return vectors[1:-1], tokens[1:-1]

@app.route('/embeddings', methods=["POST"])
def hello_world():
    utterances = request.form.get('utterances', ",keywords=")
    utterances, keywords = utterances.split(",keywords= ")

    utterances = utterances.split("<break>")
    keywords = keywords.split()
    if not utterances or not keywords:
        return jsonify({"cluster1":[], "cluster2":[]})
    keywords_tmp = []
    for k in keywords:
        keywords_tmp += k.split("_")
    keywords = set(keywords_tmp)
    vectors = []
    tokens = []
    for utterance in utterances:
        if len(utterance) < 5:
            continue
        if "<s>" not in utterance:
            utterance = "<s> " + utterance
        if "</s>" not in utterance:
            utterance = utterance + "</s>"
        u_vectors, u_tokens = get_vectors(utterance)
        vectors += u_vectors
        tokens += u_tokens

    keep_tokens = []
    keep_vectors = []
    for i, v in enumerate(vectors):
        if tokens[i] in keywords:
            keep_vectors.append(vectors[i])
            keep_tokens.append(tokens[i])
    vectors = keep_vectors
    tokens = keep_tokens
    if len(vectors) < 3:
        return jsonify({"cluster1":[], "cluster2":[]})
    kmeans = KMeans(init="random", n_clusters=3)
    labels = kmeans.fit_predict(vectors)
    clusters = [[], [], []]
    for i, label in enumerate(labels):
        if tokens[i] in keywords:
            clusters[label].append(tokens[i])
    return jsonify({"cluster1":" ".join(set(clusters[0])), "cluster2":" ".join(set(clusters[1])), "cluster3":" ".join(set(clusters[2]))})

