import arxiv
import pandas as pd
from tqdm import tqdm
import argparse
import utils

#---------------------------------------------------------------------#
# query_keyword: If there are multiple queries, they must be entered  #
# according to the following rule: query_keyword1|query_keyword2|...  #
#---------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="config")

parser.add_argument("--query_keyword", 
                    required=True)
parser.add_argument("--max_results",
                    default=10000)

args = parser.parse_args()
max_results = int(args.max_results)
print(args.query_keyword)
if "|" in args.query_keyword:
    keywords = args.query_keyword.split('|')
    query_keywords = []
    for keyword in keywords:
        query_keywords.append(f"\"{keyword}\"")
else: 
    query_keywords = [args.query_keyword]


# reuse a client with increased number of retries (3 -> 10) and increased page size (100->500).
client = arxiv.Client(num_retries=20, page_size=500)

def query_with_keywords(query):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )
    terms = []
    titles = []
    abstracts = []
    for res in tqdm(client.results(search), desc=query):
        if res.primary_category in ["cs.CV", "stat.ML", "cs.LG"]:
            terms.append(res.categories)
            titles.append(res.title)
            abstracts.append(res.summary)
    return terms, titles, abstracts


all_titles = []
all_summaries = []
all_terms = []

for query in query_keywords:
    terms, titles, abstracts = query_with_keywords(query)
    all_titles.extend(titles)
    all_summaries.extend(abstracts)
    all_terms.extend(terms)
    
    
data = pd.DataFrame({
    'titles': all_titles,
    'summaries': all_summaries,
    'terms': all_terms
})

# save data
utils.save_json(data=data, query_name=args.query_keyword)