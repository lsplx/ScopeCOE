import json
import requests
import json
import numpy as np
import random, math
import os
import json, tqdm, requests
import yaml
# from models.models import *
from openai import OpenAI
import os
import re
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def ask_LLM(question, snippet_string, url, apikey):
    system_prompt = '''You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer and it is not possible to answer based on your existing knowledge, you will generate ’I can not answer the question because of the insufficient information in documents.‘If the information in the documents is insufficient, but you can answer the question based on your existing knowledge, then proceed to answer using your knowledge. If there are inconsistencies with the facts in some of the documents, please generate the response 'There are factual errors in the provided documents.' and provide the correct answer.  ''' 
    user_text = "Document:\n" +  snippet_string +  "\n\nQuestion:\n" + question
    return getdata(system_prompt, user_text,url,apikey)



def check(question, Reference_answer,Model_response,  url, apikey):
    prompt = '''You are CompareGPT, a machine to verify the correctness of predictions. Answer with only yes/no.

You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question. All information in the ground-truth answer must be present in the prediction, including numbers and dates. You must answer "no" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.

Question: {Question}
Ground-truth answer: {Reference_answer}
Prediction:  {Model_response}

CompareGPT response:
    '''
    text2 = prompt.format(Question=question,Reference_answer=Reference_answer, Model_response =Model_response)
    return getdata_judge(text2,url,apikey)

def append_to_json_file(item, filename):
    try:
        with open(filename, 'r+') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
            data.append(item)
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()
    except FileNotFoundError:
        with open(filename, 'w') as file:
            json.dump([item], file, indent=4)

def extract_json_content(s):
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1:
        return s[start:end+1]
    return None
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def getdata(system_prompt, user_text,url,API_KEY):
    client = OpenAI(api_key=API_KEY)
    #gpt-3.5-turbo
    #gpt-4-turbo
    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    temperature= 0.1,
    messages=[
        {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_text,
                }
    ]
    )
    response = completion.choices[0].message.content

    return response
    
def getdata_judge(text,url,API_KEY):
    client = OpenAI(api_key=API_KEY)
    completion = client.chat.completions.create(
    model="gpt-4o",
    temperature= 0.1,
    messages=[
        {"role": "user", "content": text}
    ]
    )
    response = completion.choices[0].message.content

    return response


def main():
    input_file = '/HotpotQA/merge_knowledge_loop_1000_searchrank_unrevelent_conflict_new.json'
    output_file = '/HotpotQA/merge_knowledge_loop_1000_RQ3_log_wrongP25_GPT4.json'
    unlogic_file = '/HotpotQA/merge_knowledge_loop_1000_unlogknowledge_loop.json'
    url = "https://api.openai.com/v1/completions"
    api_key = "XXX"
    data = read_jsonl(input_file)
    unlogic_file = read_jsonl(unlogic_file)
    correct_num = 0
    wrong_num = 0
    for num, each in enumerate(data):
        print(num)
        # if num < 477:
        #     continue
        question = each["question"]
        knowledge = each["knowledge"]
        answer = each["right_answer"]
        conflict_knowledge = each["conflict_knowlledge"]
        answer_sentence = ""
        unlog_sentence = ""
        for eachone in unlogic_file:
            if eachone["question"] == question:
                unlog_sentence = eachone["unlogic_knowledge"]
                # unlog_sentence = eachone["unlogic_knowledge_subtitute"]
                break
        knowledge_sentences = re.split(r'[.!?]+', knowledge)
        knowledge_sentences = [x for x in knowledge_sentences if x != '']
        knowledge_len = len(knowledge_sentences)
        wrong_ratio = 0.25
        total_len = int(knowledge_len/(1-wrong_ratio))
        if total_len == knowledge_len:
            wrong_len = 1
        else:
            wrong_len = total_len - knowledge_len
        if len(conflict_knowledge) < wrong_len:
            conflict_knowledge = conflict_knowledge * 5
        string_conflict_knowledge = ' '.join(conflict_knowledge[:wrong_len])
        snippet_string = knowledge + " "  + string_conflict_knowledge
        response = ask_LLM(question, snippet_string, url, api_key)
        each["RAG_response"] = response
        is_correct = check(question, answer,response, url, api_key)
        if "yes" in is_correct.lower():
            correct_num += 1
            each["label"] = "correct"
        else:
            wrong_num += 1
            each["label"] = "wrong"
        append_to_json_file(each, output_file)
    correct_ratio = correct_num / len(data)
    wrong_ratio = wrong_num / len(data)
    print("correct_num: " + str(correct_num))
    print("wrong_num: " + str(wrong_num))
    print("correct_ratio: " + str(correct_ratio))
    print("wrong_ratio: " + str(wrong_ratio))
    print("RQ3")
    print(output_file)
if __name__ == "__main__":
    main()