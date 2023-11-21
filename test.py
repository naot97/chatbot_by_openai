# import pandas as pd
# from nltk.metrics.scores import f_measure, recall
# from utils import normalize_text
# import ast

# def f1_score(gold: str, pred: str) -> float:
#     ret = f_measure(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
#     if ret is None:  # answer is the empty string after normalizing
#         return 0.0

#     return ret

# def recall_score(gold: str, pred: str) -> float:
#     ret = recall(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
#     if ret is None:  # answer is the empty string after normalizing
#         return 0.0

#     return ret

# df = pd.read_csv("result.csv", encoding="utf-8")
# result = df.to_dict("records")

# search_time = list(df["search_time"])
# answering_time = list(df["answering_time"])

# average_search_time = sum(search_time)/len(search_time)
# average_answering_time = sum(answering_time)/len(answering_time)

# print('average search time: ', average_search_time)
# print('average answering time: ', average_answering_time)
# cost = [ast.literal_eval(record['usage'])['cost'] for record in result]
# print('sum cost: ', sum(cost)/len(search_time)) 

# recall_s = [recall_score(record['reference'], record['answer']) for record in result]
# f1_s = [f1_score(record['reference'], record['answer']) for record in result]
# print('recall: ', sum(recall_s) /len(recall_s))
# print('f1 score: ', sum(f1_s) /len(f1_s))

import requests

response = requests.post("http://118.69.66.165:5001/", json={
'text': ['Hello world', "My name's Peter", 'The sun is shining'],
'dest_language': 'vi'
})

response.json()