# ex-GPt: An Extractive-Abstractive Summarization Framework with a Sentence Embeddings Twist

We propose a summarization pipeline to pro- duce abstractive summaries of news articles. We first perform an extractive step at sentence level, in order to filter the most relevant sen- tences in the article. We use this extractive summary as conditioning to fine-tune a GPT- 2 model to perform a further abstractive step. Furthermore, we investigate on the shortcom- ings of the ROUGE metric and propose an al- ternative summarization evaluation metric and extractor model relying on sentence embed- dings. We show that using sentence embed- dings similarity measure in the extraction step captures richer latent content and can lead to improved ROUGE scores. Finally, we also show that our pipeline produces coherent and fluent end-to-end summaries.
