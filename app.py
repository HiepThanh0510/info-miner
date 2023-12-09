from transformers import AutoModel
import utils 
import argparse

#-------------------------------------------------------------------------#
# load pretrained model for encoding text to vector (features extraction) #
# - small version: 33 million parameters (based on BERT architecture)     #
# - input: text                                                           #
# - output: vector with dims = (512,)                                     #
#-------------------------------------------------------------------------#
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-small-en', 
                                  trust_remote_code=True) # trust_remote_code is needed to use the encode method

parser = argparse.ArgumentParser(description="config")

parser.add_argument("--query_keyword", 
                    type=str,
                    help='query keyword',
                    required=True)

# retrieving the relevant documents for each query provided by the user. 
parser.add_argument("--prompt", 
                    type=str,
                    help="prompt",
                    default="no-prompt") 

parser.add_argument("--document_path", 
                    type=str,
                    help='path of document',
                    default="no-document_path") 

parser.add_argument("--num_queries",
                    type=str,
                    help="number of queries",
                    default=5)

args = parser.parse_args()

if "|" in args.query_keyword:
    keywords = args.query_keyword.split('|')
    query_keywords = []
    for keyword in keywords:
        query_keywords.append(f"{keyword}") # f"\"{keyword}\""
else: 
    query_keywords = [args.query_keyword]


data_json = utils.load_one_csv(query_keywords[0]) if len(query_keywords) == 1 else utils.load_many_csv(list_name=query_keywords)

#----------------------#
# for task: user query #
#----------------------#
if args.prompt != "no-prompt":
    prompt = args.prompt
    vector_retrieval = model.encode(prompt)

#-------------------------------------------------------------------------#
# for task: extract information from document, then query relevant papers #
#-------------------------------------------------------------------------#
elif args.document_path != "no-document_path":
    all_text = utils.read_pdf(args.document_path)
    vector_retrieval = utils.document2vector(model=model,
                                             all_text=all_text)


list_score = []
for i in range(len(data_json)):
    similarity = utils.cos_sim(vector_retrieval, data_json[i]['vector_encoding'])
    list_score.append((i, similarity)) # (index, corresponding score)
    list_score_sort = sorted(list_score, key=lambda x: x[1], reverse=True)

k = int(args.num_queries)
top_k = list_score_sort[:k]
top_k_index = [i[0] for i in top_k]

for i in top_k_index:
    print(data_json[i]['summaries'])
    print('titile: ', data_json[i]['titles'])
    print('\n')