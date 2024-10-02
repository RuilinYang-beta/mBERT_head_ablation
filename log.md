* What to do? 
  - read the code of the paper "Are sixteen heads really better than one?", ask GPT to understand the head ablation procedure
  - how to do head ablation? 
  - what is the MultiNLI / XNLI data look like? 
  - how to do fine-tuning? 
  - how computationally expensive are these? 
  - how to rent computing power? 
  - how to analyze the stats? 

* Progress
  - [done] connect with SSH 
  - [done] Remote-SSH VS Code plugin 
  - [done] run model from local model on remote
  - [done] fine-tune model on remote 
  - [done] understanding progress bar STEP / TOTAL_STEP
      where TOTAL_STEP is num_epoch * num_batch_per_epoch 
  - [done] ~~About translation~~ 
    - even if I just translate the first 5,000 examples, which consists of 865,497 characters, it will cost me 110 dollars ((865497 - 500000) / 1000000 * 20 * 15) 
    - so let's use the XNLI data; but slice 1000 examples from the test set as the dev set
  - [done] hyperparam tuning reduce overfitting 
    - lr_scheduler: linearly decrease learning rate (default)
    - learning rate: 2e-5 to 5e-5
    - warmup ratio: 0.1 to 0.3 
    - weight decay: 1e-5 and 1e-2 
    - Early Stopping
  - [done] narrow down hyperparam space
  - [?] analyze data
    - in total 8 configurations: 
      - 2 options of fine-tuning data {en_only, all_lang}
      - 2 options of abalation method {head_ablation, layer_ablation}
      - 2 evaluation matrics {accuracy, f1_score}
    - under each configuration, there is a matrix of 16 * 144 size
      - 16 for 16 languages: 15 lang + all lang
      - 144 for 144 ways of ablation 
    - clustering analysis: {k-means clustering, hierarchical clustering}
    - correlation analysis: {pearson correlation, spearman correlation} 
    - dimensionality reduction: {PCA, t-SNE or UMAP}
  - theoretical and emperial baseline
    - theoretical: computed from data
    - emperical: randomly attach N HeadForClassification to mBERT and compute