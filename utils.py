from numpy.linalg import norm

import json 
import csv 
import pandas as pd 
import ast 
import numpy as np 
import PyPDF2
import os 

def cos_sim(a, b):
  similarity = (a @ b.T) / (norm(a) * norm(b))
  return similarity

def save_json(data, query_name):
  data_path = f"data/arxiv_data_{query_name}.json"
  data.to_json(data_path, orient='records')
  
def load_json(query_name):
  data_path = f"data/arxiv_data_{query_name}.json"
  with open(data_path, 'r') as file:
    data_json = json.load(file)
  return data_json
  
def save_csv(data_json, query_name):
  data_path = f"data/arxiv_data_{query_name}.csv"
  with open(data_path, 'w', newline='') as csv_file:
    # create a CSV writer
    csv_writer = csv.writer(csv_file)

    # write the header
    csv_writer.writerow(data_json[0].keys())

    # write the data
    for row in data_json:
        csv_writer.writerow(row.values())

def load_one_csv(query_name):
  data_path = f"data/arxiv_data_{query_name}.csv"
  with open(data_path, 'r', newline='') as csv_file:
    # create a CSV reader
    csv_reader = csv.DictReader(csv_file)

    # convert CSV data to a list of dictionaries
    data_json = [row for row in csv_reader]     
  
  for i in range(len(data_json)):
    data_json[i]['vector_encoding'] = np.array(ast.literal_eval(data_json[i]['vector_encoding'])) 
  
  return data_json 

def load_many_csv(list_name):
  list_csv = []
  for query_name in list_name:
    list_csv.append(f"data/arxiv_data_{query_name}.csv")
  
  all_data_json = []
  for csv_path in list_csv:
    with open(csv_path, 'r', newline='') as csv_file:
      # create a CSV reader
      csv_reader = csv.DictReader(csv_file)

      # convert CSV data to a list of dictionaries
      data_json = [row for row in csv_reader]

    for i in range(len(data_json)):
      data_json[i]['vector_encoding'] = np.array(ast.literal_eval(data_json[i]['vector_encoding']))

      # append
      all_data_json.append(data_json[i])
  return all_data_json


def read_pdf(file_name):
  file_path = f"documents/{file_name}"
  pdf_file = open(file_path, "rb")
  reader = PyPDF2.PdfReader(pdf_file)
  all_text = ""
  # extract text from all pages
  for page in range(len(reader.pages)):
      page_object = reader.pages[page]
      text = page_object.extract_text()
      all_text += text

  # close the PDF file
  pdf_file.close()
  return all_text 

def read_txt(file_name):
  data_path = f"paper/{file_name}"
  with open(data_path, 'r') as file:
    all_text = file.read()
    return all_text 


def document2vector(model, all_text):
  if len(all_text) < 2048:
    return model.encode(all_text)
  else: 
    a = len(all_text)
    b = a // 2048
    
    all_text = all_text[:b*2048]
    sub_text = [all_text[i: i+2048] for i in range(0, len(all_text), 2048)]
    sub_vectors = [model.encode(text) for text in sub_text]
    mean_vector = np.mean(sub_vectors, axis=0)
    
    return mean_vector 

#-------------------------------------#
# implement get_file_format function  #
#-------------------------------------#
def get_file_format(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower()