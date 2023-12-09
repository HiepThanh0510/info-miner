# paper-recommendation-system

# Introduction 
In this project, my goal is to build a paper recommendation system. The reason I undertake this project is that when you read specific documents and want to search for related papers to explore, or when you have an idea and want to find relevant papers for further research. That is the reason why this project came into existence.

# Set up
__Python3.8.x__ and __Pytorch 2.0.1__ are used in this repository.

+ Clone this repo:
```bash
git clone https://github.com/HiepThanh0510/info-miner
cd info-miner
```
+ To install required packages, use pip:
```bash
pip install -r requirements.txt
```
# Repository hierarchy

```
.
├── app.py
├── crawl_data.py
├── data
├── feature_extraction.py
├── documents
├── README.md
├── requirements.txt
├── test.ipynb
└── utils.py
```

# Dataset 
+ To be flexible in searching for related papers within a specific topic, I crawl data from arXiv across various topics such as image segmentation, time series, large language models, physics, and more.
+ Please review the [crawl_data.py](crawl_data.py) file to see how the data is crawled, the crawled data will be saved in json format.
+ The information of a paper that I scrape from arXiv will be in the following format
    <details>
        <summary>Data format</summary>

    ```json
        {
            "titles": "Series2Vec: Similarity-based Self-supervised Representation Learning for Time Series Classification",
            "summaries": """We argue that time series analysis is fundamentally different in nature to
            either vision or natural language processing with respect to the forms of
            meaningful self-supervised learning tasks that can be defined. Motivated by
            this insight, we introduce a novel approach called \textit{Series2Vec} for
            self-supervised representation learning. Unlike other self-supervised methods
            in time series, which carry the risk of positive sample variants being less
            similar to the anchor sample than series in the negative set, Series2Vec is
            trained to predict the similarity between two series in both temporal and
            spectral domains through a self-supervised task. Series2Vec relies primarily on
            the consistency of the unsupervised similarity step, rather than the intrinsic
            quality of the similarity measurement, without the need for hand-crafted data
            augmentation. To further enforce the network to learn similar representations
            for similar time series, we propose a novel approach that applies
            order-invariant attention to each representation within the batch during
            training. Our evaluation of Series2Vec on nine large real-world datasets, along
            with the UCR/UEA archive, shows enhanced performance compared to current
            state-of-the-art self-supervised techniques for time series. Additionally, our
            extensive experiments show that Series2Vec performs comparably with fully
            supervised training and offers high efficiency in datasets with limited-labeled
            data. Finally, we show that the fusion of Series2Vec with other representation
            learning models leads to enhanced performance for time series classification.
            Code and models are open-source at \url{https://github.com/Navidfoumani/Series2Vec.}""",
            "terms": "['cs.LG']"
        }
    ```
    </details>
+ How to run the crawl_data.py file to crawl papers from arXiv: 
    ```python
    python crawl_data.py --query_keyword "topic_you_want_to_crawl" --max_results 20000
   
    ```
+ Example: 
    ```python
    python crawl_data.py --query_keyword "physics"  --max_results 20000
    ```

# Feature extraction 
+ In this preprocessing step, I utilize the pretrained model jina-embeddings-v2-small-en from the Finetuner team, Jina AI. Using this model to encode the abstracts of papers, the encoded vectors will be saved alongside the original data in CSV format.
+ Please take a look at the [feature_extraction.py](feature_extraction.py) file for insights into extracting features from the abstracts of papers. 
+ To perform this feature extraction process for a specific topic, follow this step:
    ```python
    python feature_extraction.py --query_keyword "topics_you_want_to_do_feature_extraction"
    ```
+ Example: 
    + Executing with 1 specific topic:
        ```python
        python feature_extraction.py --query_keyword "physics"
        ```
    + Executing with multiple topics: 
        ```python
        python feature_extraction.py --query_keyword "physics|time_series|transformers"
        ```

# Inference
