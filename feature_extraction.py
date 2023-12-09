from transformers import AutoModel
import utils 
import argparse

parser = argparse.ArgumentParser(description="config")

parser.add_argument("--query_keyword", 
                    required=True)

args = parser.parse_args()
query_keyword = args.query_keyword

#-------------------------------------------------------------------------#
# load pretrained model for encoding text to vector (features extraction) #
# - small version: 33 million parameters (based on BERT architecture)     #
# - input: text                                                           #
# - output: vector with dims = (512,)                                     #
#-------------------------------------------------------------------------#
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-small-en', 
                                  trust_remote_code=True) # trust_remote_code is needed to use the encode method


#--------------------------#
# load data (json format)  #
#--------------------------#
data_json = utils.load_json(query_name=query_keyword)


#------------------#
# extraction stage #
#------------------#
for i in range(len(data_json)):
    data_json[i]['vector_encoding'] = str(list(model.encode(data_json[i]['summaries'])))


#------------------------#
# save data (csv format) #
#------------------------#
utils.save_csv(data_json=data_json,
               query_name=query_keyword)