# overview 

code for rebuttals. for some code like add oracles will just change code in src(maybe), but results will all under rebuttal folder. 



## code list 
1. add P value for all modalities-tissue pair for MUGO against other methods. 
raw script to generate benchmark table: /home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/benchmarking_on_geneset.py
/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/get_new_table2.py



2. for N>>1000 time/space effactive curve. 
写一个 for 循环，把 sequence length $N$ 从 1000 设到 10000（甚至更长），分别跑一遍 MUGO 的前向和反向传播。记录 time.time() 和 torch.cuda.max_memory_allocated()。用 matplotlib 画个折线图，证明你们的方法是 $O(N)$ 线性增长的。

3. multi-random seed for MUGO 
给你的 MUGO 加个 argparse 传入 seed (比如 42, 43, 44, 45, 46)，写个 bash 脚本 for i in {42..46}; do python train.py --seed $i; done。跑完后算个 mean 和 std，加到表格的括号里（例如：$0.85 \pm 0.02$）。

4. add foundation model
如果你有现成的 Enformer 或者 scGPT 的 wrapper，直接换个 backbone 跑一下你最小的那个 dataset。不用跑全，只要证明“MUGO 框架是 model-agnostic（模型无关）的”即可。

5. benchmark GA, Greedy
对策： 挑一个搜索空间不那么夸张的 locus（基因座）。自己手写一个极其粗糙的 遗传算法 (GA) 和 贪心搜索 (Greedy)。你的目标不是把 GA 写得多好，而是让 GA 跑得慢且容易陷入局部最优，从而衬托出 MUGO 连续松弛的丝滑与高效。

6. Causal Inference 与 CADD/FunSeq 不公平对比 (R3 W1, W3)
话术对策： 承认 CADD/FunSeq 和咱们的优化目标不一样（一个是纯打分，一个是组合优化）。但要强调：“在目前的生物信息学界，没有任何一个现成的工具在解决连续组合变异的问题，所以 CADD/FunSeq 是我们能找到的最接近的、被广泛认可的 Baseline。同时为了响应您的要求，我们补充了相同优化目标下的 GA 和 Greedy 算法对比。”

7. mutagenesis/MPRA/CRispy and wetlab validation 
TODO 

