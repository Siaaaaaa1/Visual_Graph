0. 数据预处理（只需一次）

python datasets/add_sim/preprocess_embeddings.py

1. 蒸馏

bash distill/run_distill_pipeline.sh

2. SFT（三个数据集，按需选择）

bash distill/run_sft_cora.sh
bash distill/run_sft_pubmed.sh
bash distill/run_sft_arxiv.sh

3. RL（三个数据集，按需选择）

bash Our_examples/run_cora_Graph_4B_Thinking.sh
bash Our_examples/run_pubmed_Graph_4B_Thinking.sh
bash Our_examples/run_arxiv_Graph_4B_Thinking.sh