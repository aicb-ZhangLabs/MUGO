'''
3. 针对 ICIP 的作图/展示建议
不用把 3000 个结果都贴出来，ICIP 的 Paper 篇幅有限，你需要一张统计图 + 一个 Case Study。

图 1：总体统计 (Bar Chart 或 Box Plot)
展示那 ~1000 多个有效基因的汇总数据。

Metric 1: Enrichment (富集度)

你的 Top-10 SNP 里包含真实 eQTL 的比例，对比“随机抽取 10 个 SNP”包含 eQTL 的比例。

肯定会高出几十倍，这个图会很好看。

Metric 2: Top-k Precision

Top-1 Accuracy: ?%

Top-5 Accuracy: ?%

图 2：Case Study (就是 ZNF263)
把 ZNF263 作为一个具体的例子画出来。

画一条基因组坐标轴。

上面画 GTEx 的 P-value（倒置的 Manhattan Plot 风格）。

下面画你模型的 Score。

高亮重合的部分： 展示你的高分峰值和 GTEx 的高显著性峰值是对齐的。
'''