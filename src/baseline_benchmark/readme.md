# overview 
this use to benchmark some baseline methods. Choose 500 genes for 6 tracks and just benchmark borzoi can be good(best model)

## methods to benchmark

Class1： 核心对手 

1. Greedy ISM (Top-K Single Scan) :  暴力扫描序列里的每一个SNP（比如2000个），计算每个SNP单独突变后的 Borzoi Score 变化。然后简单地把分数最高的 Top K 个挑出来。 additive assumption。 

2. Random Search (Baseline of Baselines)： 从 Candidate Pool 里随机抽取 K 个 SNP，重复 N 次（比如1000次），取分最高的那个组合。（enrichment 1x） 

 

Class2： 速度/效率陪跑 (The "Fast Proxy") 

3. Saliency Map (Gradient-based)： 计算 Output 对 Input 的梯度 ($\nabla x$)。直接取梯度绝对值最大的 Top K 个位点。 

4. Feature Ablation (Masking)： 类似于 ISM，但是把位点 Mask 成 0 或者 N，而不是突变成别的碱基。 

 

Class3： 生物学方法/数据：不用跑代码可以直接下载数据 

5. FunSeq2 / 6. CADD / 7. DeepSEA Score：  这些是现成的、基于规则或旧模型的致病性打分工具。把你 Candidate Pool 里的 2000 个 SNP 的 rsID 拿去查一下这些分数，选出topK，证明有些我找到的高分SNP被这些method漏掉了。 

## metric used to evlauate dfferent method 
1. gene expression gain. (Funseq etc traditional method not have this score)
2. enrichment for GTEx hit, conservative loci hit, GWAS catelog disease hit. 


