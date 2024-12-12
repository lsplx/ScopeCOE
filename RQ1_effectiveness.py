import json
import requests
import json
import numpy as np
import random, math
import os
import json, tqdm, requests
from openai import OpenAI
import os
import time
import openai
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

def ask_LLM(question, snippet_string, url, apikey):
    system_prompt = '''You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer and it is not possible to answer based on your existing knowledge, you will generate ’I can not answer the question because of the insufficient information in documents.‘If the information in the documents is insufficient, but you can answer the question based on your existing knowledge, then proceed to answer using your knowledge. If there are inconsistencies with the facts in some of the documents, please generate the response 'There are factual errors in the provided documents.' and provide the correct answer.  ''' 
    user_text = "Document:\n" +  snippet_string +  "\n\nQuestion:\n" + question
    return getdata(system_prompt, user_text,url,apikey)

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
    try:
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
    except openai.APITimeoutError:
        time.sleep(10)
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
    try:
        completion = client.chat.completions.create(
        model="gpt-4o",
        temperature= 0.1,
        messages=[
            {"role": "user", "content": text}
        ]
        )
    except openai.APITimeoutError:
        time.sleep(10)
        completion = client.chat.completions.create(
        model="gpt-4o",
        temperature= 0.1,
        messages=[
            {"role": "user", "content": text}
        ]
        )
    except openai.NotFoundError:
        time.sleep(10)
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
    input_file = '/data/2WikiMultihopQA_CoE.json'
    output_file = '/data/2WikiMultihopQA_CoE_RQ1_p75_unlog_GPT4.json'
    url = "https://api.openai.com/v1/completions"
    api_key = "XXX"
    data = read_jsonl(input_file)
    correct_num = 0
    wrong_num = 0
    for num, each in enumerate(data):
        print(num)
        knowledge = each["knowledge"]
        question = each['CoE']
        # sub_unlog_knowledge = each["unlogic_knowledge_subtitute"]
        unlogic_knowledge = each['Senp_Non_CoE']
        answer = each["answer"]
        unrelevent_snippet = each["irrelevant_info"]
        #knowledge count rate
        ratio = 0.75
        knowledge_len = len(unlogic_knowledge)
        limit_len = (knowledge_len / ratio) - knowledge_len
        unrelevent_all = " ".join(unrelevent_snippet)
        if len(unrelevent_all) < limit_len:
            unrelevent_all = unrelevent_all * 5
        limit_unrelevent_snippet = unrelevent_all[:int(limit_len)] 
        snippet_string = unlogic_knowledge  + " " + limit_unrelevent_snippet
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
    total_num = correct_num + wrong_num
    correct_ratio = correct_num / total_num
    wrong_ratio = wrong_num / total_num
    print("correct_num: " + str(correct_num))
    print("wrong_num: " + str(wrong_num))
    print("corect_ratio: " + str(correct_ratio))
    print("wrong_ratio: " + str(wrong_ratio))
    print("RQ1")
    print(output_file)
if __name__ == "__main__":
    main()
    