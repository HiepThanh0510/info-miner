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

    ```
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
Please review the file [app.py](app.py) for insights into the query process.
### Retrieving the relevant documents for each query provided by the user. 
+ To perform this task, proceed as follows:
    ```python
    python app.py --query_keyword "topics_you_want_to_search" --prompt "your prompt" --num_queries 10
    ```
+ Example: 
    ```python
    python app.py --query_keyword "image_segmentation|time_series|physics|computation_and_language" --prompt "i am researching image segmentation" --num_queries 3
    ```

     <details>
        <summary>Queries result</summary>

    ```
        Top 1:
            Titile:  Beyond Pixels: A Comprehensive Survey from Bottom-up to Semantic Image Segmentation and Cosegmentation
            Abstract: Image segmentation refers to the process to divide an image into
            nonoverlapping meaningful regions according to human perception, which has
            become a classic topic since the early ages of computer vision. A lot of
            research has been conducted and has resulted in many applications. However,
            while many segmentation algorithms exist, yet there are only a few sparse and
            outdated summarizations available, an overview of the recent achievements and
            issues is lacking. We aim to provide a comprehensive review of the recent
            progress in this field. Covering 180 publications, we give an overview of broad
            areas of segmentation topics including not only the classic bottom-up
            approaches, but also the recent development in superpixel, interactive methods,
            object proposals, semantic image parsing and image cosegmentation. In addition,
            we also review the existing influential datasets and evaluation metrics.
            Finally, we suggest some design flavors and research directions for future
            research in image segmentation.
        Top 2:
            Titile:  Visual-hint Boundary to Segment Algorithm for Image Segmentation
            Abstract: Image segmentation has been a very active research topic in image analysis
            area. Currently, most of the image segmentation algorithms are designed based
            on the idea that images are partitioned into a set of regions preserving
            homogeneous intra-regions and inhomogeneous inter-regions. However, human
            visual intuition does not always follow this pattern. A new image segmentation
            method named Visual-Hint Boundary to Segment (VHBS) is introduced, which is
            more consistent with human perceptions. VHBS abides by two visual hint rules
            based on human perceptions: (i) the global scale boundaries tend to be the real
            boundaries of the objects; (ii) two adjacent regions with quite different
            colors or textures tend to result in the real boundaries between them. It has
            been demonstrated by experiments that, compared with traditional image
            segmentation method, VHBS has better performance and also preserves higher
            computational efficiency.
        Top 3: 
            Titile:  Multilevel Threshold Based Gray Scale Image Segmentation using Cuckoo Search
            Abstract: Image Segmentation is a technique of partitioning the original image into
            some distinct classes. Many possible solutions may be available for segmenting
            an image into a certain number of classes, each one having different quality of
            segmentation. In our proposed method, multilevel thresholding technique has
            been used for image segmentation. A new approach of Cuckoo Search (CS) is used
            for selection of optimal threshold value. In other words, the algorithm is used
            to achieve the best solution from the initial random threshold values or
            solutions and to evaluate the quality of a solution correlation function is
            used. Finally, MSE and PSNR are measured to understand the segmentation
            quality.
    ```
    </details>

### For a list of documents (which can be in pdf or txt format), we need to encode the text within the document files and use it to query and find related papers.
+ To perform this task, proceed as follows: 
    ```python
    python app.py --query_keyword "topics_you_want_to_search" --document_path "file_in_document_folder" --num_queries 5
    ```
+ Example for pdf file:
    ```python
    python app.py --query_keyword "image_segmentation|time_series|physics|computation_and_language" --document_path "resnet.pdf" --num_queries 5
    ```

    <details>
        <summary>Queries result</summary>

    ```
        get pdf file format
        Top 1:
            Titile:  Spatially Adaptive Computation Time for Residual Networks
            Abstract: This paper proposes a deep learning architecture based on Residual Network
            that dynamically adjusts the number of executed layers for the regions of the
            image. This architecture is end-to-end trainable, deterministic and
            problem-agnostic. It is therefore applicable without any modifications to a
            wide range of computer vision problems such as image classification, object
            detection and image segmentation. We present experimental results showing that
            this model improves the computational efficiency of Residual Networks on the
            challenging ImageNet classification and COCO object detection datasets.
            Additionally, we evaluate the computation time maps on the visual saliency
            dataset cat2000 and find that they correlate surprisingly well with human eye
            fixation positions.


        Top 2:
            Titile:  Augmenting Convolutional networks with attention-based aggregation
            Abstract: We show how to augment any convolutional network with an attention-based
            global map to achieve non-local reasoning. We replace the final average pooling
            by an attention-based aggregation layer akin to a single transformer block,
            that weights how the patches are involved in the classification decision. We
            plug this learned aggregation layer with a simplistic patch-based convolutional
            network parametrized by 2 parameters (width and depth). In contrast with a
            pyramidal design, this architecture family maintains the input patch resolution
            across all the layers. It yields surprisingly competitive trade-offs between
            accuracy and complexity, in particular in terms of memory consumption, as shown
            by our experiments on various computer vision tasks: object classification,
            image segmentation and detection.


        Top 3:
            Titile:  One-Shot Learning for Semantic Segmentation
            Abstract: Low-shot learning methods for image classification support learning from
            sparse data. We extend these techniques to support dense semantic image
            segmentation. Specifically, we train a network that, given a small set of
            annotated images, produces parameters for a Fully Convolutional Network (FCN).
            We use this FCN to perform dense pixel-level prediction on a test image for the
            new semantic class. Our architecture shows a 25% relative meanIoU improvement
            compared to the best baseline methods for one-shot segmentation on unseen
            classes in the PASCAL VOC 2012 dataset and is at least 3 times faster.


        Top 4:
            Titile:  High-performance Semantic Segmentation Using Very Deep Fully Convolutional Networks
            Abstract: We propose a method for high-performance semantic image segmentation (or
            semantic pixel labelling) based on very deep residual networks, which achieves
            the state-of-the-art performance. A few design factors are carefully considered
            to this end.
            We make the following contributions. (i) First, we evaluate different
            variations of a fully convolutional residual network so as to find the best
            configuration, including the number of layers, the resolution of feature maps,
            and the size of field-of-view. Our experiments show that further enlarging the
            field-of-view and increasing the resolution of feature maps are typically
            beneficial, which however inevitably leads to a higher demand for GPU memories.
            To walk around the limitation, we propose a new method to simulate a high
            resolution network with a low resolution network, which can be applied during
            training and/or testing. (ii) Second, we propose an online bootstrapping method
            for training. We demonstrate that online bootstrapping is critically important
            for achieving good accuracy. (iii) Third we apply the traditional dropout to
            some of the residual blocks, which further improves the performance. (iv)
            Finally, our method achieves the currently best mean intersection-over-union
            78.3\% on the PASCAL VOC 2012 dataset, as well as on the recent dataset
            Cityscapes.``


        Top 5:
            Titile:  Deep Learning Model with GA based Feature Selection and Context Integration
            Abstract: Deep learning models have been very successful in computer vision and image
            processing applications. Since its inception, Many top-performing methods for
            image segmentation are based on deep CNN models. However, deep CNN models fail
            to integrate global and local context alongside visual features despite having
            complex multi-layer architectures. We propose a novel three-layered deep
            learning model that assiminlate or learns independently global and local
            contextual information alongside visual features. The novelty of the proposed
            model is that One-vs-All binary class-based learners are introduced to learn
            Genetic Algorithm (GA) optimized features in the visual layer, followed by the
            contextual layer that learns global and local contexts of an image, and finally
            the third layer integrates all the information optimally to obtain the final
            class label. Stanford Background and CamVid benchmark image parsing datasets
            were used for our model evaluation, and our model shows promising results. The
            empirical analysis reveals that optimized visual features with global and local
            contextual information play a significant role to improve accuracy and produce
            stable predictions comparable to state-of-the-art deep CNN models.
    ```
    </details>

+ Example for txt file:
    ```python
    python app.py --query_keyword "image_segmentation|time_series|physics|computation_and_language" --document_path "time_series_abstract.txt" --num_queries 5
    ```

     <details>
        <summary>Queries result</summary>

    ```
        get txt file format
        Top 1:
            Titile:  Learning Robust and Consistent Time Series Representations: A Dilated Inception-Based Approach
            Abstract: Representation learning for time series has been an important research area
            for decades. Since the emergence of the foundation models, this topic has
            attracted a lot of attention in contrastive self-supervised learning, to solve
            a wide range of downstream tasks. However, there have been several challenges
            for contrastive time series processing. First, there is no work considering
            noise, which is one of the critical factors affecting the efficacy of time
            series tasks. Second, there is a lack of efficient yet lightweight encoder
            architectures that can learn informative representations robust to various
            downstream tasks. To fill in these gaps, we initiate a novel sampling strategy
            that promotes consistent representation learning with the presence of noise in
            natural time series. In addition, we propose an encoder architecture that
            utilizes dilated convolution within the Inception block to create a scalable
            and robust network architecture with a wide receptive field. Experiments
            demonstrate that our method consistently outperforms state-of-the-art methods
            in forecasting, classification, and abnormality detection tasks, e.g. ranks
            first over two-thirds of the classification UCR datasets, with only $40\%$ of
            the parameters compared to the second-best approach. Our source code for
            CoInception framework is accessible at
            https://github.com/anhduy0911/CoInception.


        Top 2:
            Titile:  NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time Series Pretraining
            Abstract: Recent research on time-series self-supervised models shows great promise in
            learning semantic representations. However, it has been limited to small-scale
            datasets, e.g., thousands of temporal sequences. In this work, we make key
            technical contributions that are tailored to the numerical properties of
            time-series data and allow the model to scale to large datasets, e.g., millions
            of temporal sequences. We adopt the Transformer architecture by first
            partitioning the input into non-overlapping windows. Each window is then
            characterized by its normalized shape and two scalar values denoting the mean
            and standard deviation within each window. To embed scalar values that may
            possess arbitrary numerical scales to high-dimensional vectors, we propose a
            numerically multi-scaled embedding module enumerating all possible scales for
            the scalar values. The model undergoes pretraining using the proposed
            numerically multi-scaled embedding with a simple contrastive objective on a
            large-scale dataset containing over a million sequences. We study its transfer
            performance on a number of univariate and multivariate classification
            benchmarks. Our method exhibits remarkable improvement against previous
            representation learning approaches and establishes the new state of the art,
            even compared with domain-specific non-learning-based methods.


        Top 3:
            Titile:  Evaluating Explanation Methods for Multivariate Time Series Classification
            Abstract: Multivariate time series classification is an important computational task
            arising in applications where data is recorded over time and over multiple
            channels. For example, a smartwatch can record the acceleration and orientation
            of a person's motion, and these signals are recorded as multivariate time
            series. We can classify this data to understand and predict human movement and
            various properties such as fitness levels. In many applications classification
            alone is not enough, we often need to classify but also understand what the
            model learns (e.g., why was a prediction given, based on what information in
            the data). The main focus of this paper is on analysing and evaluating
            explanation methods tailored to Multivariate Time Series Classification (MTSC).
            We focus on saliency-based explanation methods that can point out the most
            relevant channels and time series points for the classification decision. We
            analyse two popular and accurate multivariate time series classifiers, ROCKET
            and dResNet, as well as two popular explanation methods, SHAP and dCAM. We
            study these methods on 3 synthetic datasets and 2 real-world datasets and
            provide a quantitative and qualitative analysis of the explanations provided.
            We find that flattening the multivariate datasets by concatenating the channels
            works as well as using multivariate classifiers directly and adaptations of
            SHAP for MTSC work quite well. Additionally, we also find that the popular
            synthetic datasets we used are not suitable for time series analysis.


        Top 4:
            Titile:  TimeDRL: Disentangled Representation Learning for Multivariate Time-Series
            Abstract: Multivariate time-series data in numerous real-world applications (e.g.,
            healthcare and industry) are informative but challenging due to the lack of
            labels and high dimensionality. Recent studies in self-supervised learning have
            shown their potential in learning rich representations without relying on
            labels, yet they fall short in learning disentangled embeddings and addressing
            issues of inductive bias (e.g., transformation-invariance). To tackle these
            challenges, we propose TimeDRL, a generic multivariate time-series
            representation learning framework with disentangled dual-level embeddings.
            TimeDRL is characterized by three novel features: (i) disentangled derivation
            of timestamp-level and instance-level embeddings from patched time-series data
            using a [CLS] token strategy; (ii) utilization of timestamp-predictive and
            instance-contrastive tasks for disentangled representation learning, with the
            former optimizing timestamp-level embeddings with predictive loss, and the
            latter optimizing instance-level embeddings with contrastive loss; and (iii)
            avoidance of augmentation methods to eliminate inductive biases, such as
            transformation-invariance from cropping and masking. Comprehensive experiments
            on 6 time-series forecasting datasets and 5 time-series classification datasets
            have shown that TimeDRL consistently surpasses existing representation learning
            approaches, achieving an average improvement of forecasting by 57.98% in MSE
            and classification by 1.25% in accuracy. Furthermore, extensive ablation
            studies confirmed the relative contribution of each component in TimeDRL's
            architecture, and semi-supervised learning evaluations demonstrated its
            effectiveness in real-world scenarios, even with limited labeled data.


        Top 5:
            Titile:  TimeDRL: Disentangled Representation Learning for Multivariate Time-Series
            Abstract: Multivariate time-series data in numerous real-world applications (e.g.,
            healthcare and industry) are informative but challenging due to the lack of
            labels and high dimensionality. Recent studies in self-supervised learning have
            shown their potential in learning rich representations without relying on
            labels, yet they fall short in learning disentangled embeddings and addressing
            issues of inductive bias (e.g., transformation-invariance). To tackle these
            challenges, we propose TimeDRL, a generic multivariate time-series
            representation learning framework with disentangled dual-level embeddings.
            TimeDRL is characterized by three novel features: (i) disentangled derivation
            of timestamp-level and instance-level embeddings from patched time-series data
            using a [CLS] token strategy; (ii) utilization of timestamp-predictive and
            instance-contrastive tasks for disentangled representation learning, with the
            former optimizing timestamp-level embeddings with predictive loss, and the
            latter optimizing instance-level embeddings with contrastive loss; and (iii)
            avoidance of augmentation methods to eliminate inductive biases, such as
            transformation-invariance from cropping and masking. Comprehensive experiments
            on 6 time-series forecasting datasets and 5 time-series classification datasets
            have shown that TimeDRL consistently surpasses existing representation learning
            approaches, achieving an average improvement of forecasting by 57.98% in MSE
            and classification by 1.25% in accuracy. Furthermore, extensive ablation
            studies confirmed the relative contribution of each component in TimeDRL's
            architecture, and semi-supervised learning evaluations demonstrated its
            effectiveness in real-world scenarios, even with limited labeled data.
    ```
    </details>

# Conclusion
The above elements have demonstrated the feasibility of the project, but to increase accuracy and query more papers in the future, the tasks to be performed are:
+ Utilize a vector database.
+ Use the pretrained model jina-embeddings-v2-base-en with 137 million parameters, significantly more than the 33 million parameters of jina-embeddings-v2-small-en.
+ Implement microservices.