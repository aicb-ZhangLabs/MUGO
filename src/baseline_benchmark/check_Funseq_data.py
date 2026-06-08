import pysam
import os

# 文件路径
FILE = "/home/dongbos/Combine_optim_Borzoi_SNP/dataset/Funseq2_data/hg38.funseq2.1.6.liftover.bed.bgz"

def diagnose():
    if not os.path.exists(FILE):
        print(f"❌ File not found: {FILE}")
        return

    print(f"📂 Opening: {FILE}")
    try:
        fs = pysam.TabixFile(FILE)
        
        # 1. 检查索引里到底有哪些染色体
        print("\n🔍 Checking Index Contigs (First 10):")
        print(fs.contigs[:10]) 
        
        # 判断是用 'chr1' 还是 '1'
        has_chr1 = 'chr1' in fs.contigs
        has_1 = '1' in fs.contigs
        print(f"\n   Contains 'chr1'? {has_chr1}")
        print(f"   Contains '1'?    {has_1}")
        
        target_chrom = 'chr1' if has_chr1 else ('1' if has_1 else None)
        
        if not target_chrom:
            print("❌ Could not determine chromosome format (neither 'chr1' nor '1' found).")
            return

        print(f"\n🎯 Using query chromosome: '{target_chrom}'")

        # 2. 不带坐标，直接获取该染色体的第一条记录
        # 这能让我们看到真正的“正文”长什么样，跳过头部那些怪异的 KI...
        print(f"📖 Fetching first record for {target_chrom} (no coordinate limit)...")
        try:
            # fetch 不带 start/end 会返回该染色体所有记录
            # 我们只看第一个
            iterator = fs.fetch(target_chrom)
            first_rec = next(iterator)
            print(f"   ✅ First Record Raw: {first_rec}")
            
            parts = first_rec.split('\t')
            print(f"   🧩 Columns Analysis:")
            for i, p in enumerate(parts):
                print(f"      Col {i}: {p}")
                
            # 自动判断 Score 在哪一列
            # 通常 Score 是数字，或者是带分号的字符串
            # 你的 zcat 显示 Score 好像在 Col 6 (0-based) ? "0.253;No;..."
            # 让我们看看真正的 chr1 数据的 Score 在哪
            
        except StopIteration:
            print(f"   ⚠️ No records found for {target_chrom} (Empty chromosome?)")
            
        # 3. 再次尝试之前的失败查询 (用确认过的 Chrom)
        test_pos = 143199444 # 这是一个可能存在的点
        print(f"\n🔄 Retrying specific query at {target_chrom}:{test_pos}...")
        try:
            # 放宽范围
            recs = list(fs.fetch(target_chrom, test_pos-100, test_pos+100))
            print(f"   Found {len(recs)} records near {test_pos}.")
            if recs:
                print(f"   Example: {recs[0]}")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")

    except Exception as e:
        print(f"❌ Critical Error opening Tabix file: {e}")

if __name__ == "__main__":
    diagnose()