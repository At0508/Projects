import numpy as np
import pandas as pd
import os
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import io
import logging
import re

# Setting up the logger
logging.basicConfig(filename="processing.log",
                    format='%(asctime)s::%(message)s',
                    filemode='w')

logger = logging.getLogger()

# Initializing the Models
try:
  tokenizer_esg = BertTokenizer.from_pretrained('./ESG_Model/tokenizer_esg', num_labels=4)
  model_esg = BertForSequenceClassification.from_pretrained('./ESG_Model/model_esg')
except Exception as e:
  logger.error(f"Could not load esg model. "+str(e))
  exit()

try:
  tokenizer_sent = BertTokenizer.from_pretrained('./SENT_Model/tokenizer_sent', num_labels=3)
  model_sent = BertForSequenceClassification.from_pretrained('./SENT_Model/model_sent')
except Exception as e:
  logger.error(f"Could not load sent model. "+str(e))
  exit()


def get_ids_masks(data_lines):
  input_ids_esg = []
  attention_mask_esg = []
  input_ids_sent = []
  attention_mask_sent = []

  for data_line in data_lines:
    chunksize = 512
    tokens_esg = tokenizer_esg(data_line, add_special_tokens=False, return_tensors='pt')
    tokens_sent = tokenizer_sent(data_line, add_special_tokens=False, return_tensors='pt')

    ids_esg = tokens_esg['input_ids'][0]
    mask_esg = tokens_esg['attention_mask'][0]
    ids_sent = tokens_sent['input_ids'][0]
    mask_sent = tokens_sent['attention_mask'][0]

    ids_esg = torch.split(ids_esg, chunksize-2)
    mask_esg = torch.split(mask_esg, chunksize-2)
    ids_sent = torch.split(ids_sent, chunksize-2)
    mask_sent = torch.split(mask_sent, chunksize-2)

    if len(ids_esg) > 1:
      vec = [torch.nn.functional.pad(t, (0, chunksize - len(t))) for t in ids_esg]
      input_ids_esg.append(torch.stack(vec).long())

      vec = [torch.nn.functional.pad(t, (0, chunksize - len(t))) for t in mask_esg]
      attention_mask_esg.append(torch.stack(vec).int())

      vec = [torch.nn.functional.pad(t, (0, chunksize - len(t))) for t in ids_sent]
      input_ids_sent.append(torch.stack(vec).long())

      vec = [torch.nn.functional.pad(t, (0, chunksize - len(t))) for t in mask_sent]
      attention_mask_sent.append(torch.stack(vec).int())
    
    elif len(ids_esg[0]) == 0:
      continue
    else:
      input_ids_esg.append(torch.stack(ids_esg).long())
      attention_mask_esg.append(torch.stack(mask_esg).int())
      input_ids_sent.append(torch.stack(ids_sent).long())
      attention_mask_sent.append(torch.stack(mask_sent).int())

  return input_ids_esg, attention_mask_esg, input_ids_sent, attention_mask_sent

folderLocation = "./input-txts"
resultLocation = "./Results"
folderList = os.listdir(folderLocation)
folderList.sort()

# Define the range for processing folders
from_index = 0  # Change this value
to_index = 10    # Change this value
folderList = folderList[from_index:to_index]

print("Processing Folders from index", from_index, "to", to_index)

for folderName in folderList:
  fileLocation = f"{folderLocation}/{folderName}"
  filesList = os.listdir(fileLocation)
  filesList.sort()
  print(folderName)

  final_df = pd.DataFrame()

  for file_name in filesList:
    try:
      with open(fileLocation + '/' + file_name, 'r', encoding='utf-8') as sampleFile:
        rawData = sampleFile.read()
    except UnicodeDecodeError:
      try:
        with open(fileLocation + '/' + file_name, 'r', encoding='cp1252') as sampleFile:
          rawData = sampleFile.read()
      except:
        print(f"Could not open/read file {folderName}/{file_name}")
        continue
    
    processedData = rawData.replace('\n', ' ')
    data_lines = processedData.split('. ')

    if len(data_lines) < 10:
      logger.error(f"File empty/not enough data {folderName}/{file_name}")
      continue

    try:
      input_ids_esg, attention_mask_esg, input_ids_sent, attention_mask_sent = get_ids_masks(data_lines)
    except Exception as e:
      logger.error(f"Could not create tokenized ids for {folderName}/{file_name}")
      continue

    year = re.findall('\d\d-\d\d', file_name)[0]
    print("Year:%s Lines:%d" % (year, len(input_ids_esg)))

    if len(input_ids_esg) == 0:
      logger.error(f"This file could not read properly. No text ids. File: {folderName}/{file_name}")
      continue

    esg_results = []
    sent_results = []

    n = len(input_ids_esg)

    with torch.no_grad():
      for i in range(n):
        try:
          op = model_esg(input_ids=input_ids_esg[i], attention_mask=attention_mask_esg[i])
          prob = torch.nn.functional.softmax(op.logits, dim=1)
          avg_prob = prob.mean(dim=0)
          esg_results.append(avg_prob)
        except Exception as e:
          logger.error(f"Error processing ESG for {folderName}/{file_name}. " + str(e))
          continue

      for i in range(n):
        try:
          op_sent = model_sent(input_ids=input_ids_sent[i], attention_mask=attention_mask_sent[i])
          prob_sent = torch.nn.functional.softmax(op_sent.logits, dim=1)
          avg_prob = prob_sent.mean(dim=0)
          sent_results.append(avg_prob)
        except Exception as e:
          logger.error(f"Error processing SENT for {folderName}/{file_name}. " + str(e))
          continue

    try:
      esg_results = torch.stack(esg_results).cpu().detach().numpy()
      sent_results = torch.stack(sent_results).cpu().detach().numpy()
    except Exception as e:
      logger.error(f"Could not convert tensors to numpy for {folderName}/{file_name}. " + str(e))
      continue
    
    try:
      df = pd.concat([pd.DataFrame(esg_results, columns=["None", "Environmental", "Social", "Governance"]), 
                      pd.DataFrame(sent_results, columns=["Neutral", "Positive", "Negative"])], axis=1)
      df.index = [year for _ in range(len(df))]
      final_df = pd.concat([final_df, df], axis=0)
    except Exception as e:
      logger.error(f"Error creating dataframe for {folderName}/{file_name}. " + str(e))

  final_df.to_csv(f"{resultLocation}/{folderName}.csv")
