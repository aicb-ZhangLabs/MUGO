# overview
Benchmark baseline methods on the Borzoi setting using a fixed gene and track set.

## methods to benchmark

Class1： 核心对手 

1. Greedy ISM (Top-K Single Scan): 扫描序列里的每一个候选 SNP，计算每个 SNP 单独突变后的 Borzoi Score 变化，并按单点边际效应选择 Top K。该方法对应 additive assumption。

2. Random Search: 从 Candidate Pool 里随机抽取 K 个 SNP，重复 N 次，并报告 trials 中得分最高的组合。（enrichment 1x）

 

Class2： 速度/效率陪跑 (The "Fast Proxy") 

3. Saliency Map (Gradient-based): 计算 Output 对 Input 的梯度 ($\nabla x$)，并按梯度绝对值选择 Top K 位点。

4. Feature Ablation (Masking)： 类似于 ISM，但是把位点 Mask 成 0 或者 N，而不是突变成别的碱基。 

 

Class3： 生物学方法/数据：不用跑代码可以直接下载数据 

5. FunSeq2 / 6. CADD / 7. DeepSEA Score：  这些是现成的、基于规则或旧模型的致病性打分工具。把你 Candidate Pool 里的 2000 个 SNP 的 rsID 拿去查一下这些分数，选出topK，证明有些我找到的高分SNP被这些method漏掉了。 

## metric used to evlauate dfferent method 
1. gene expression gain. (Funseq etc traditional method not have this score)
2. enrichment for GTEx hit, conservative loci hit, GWAS catelog disease hit. 

