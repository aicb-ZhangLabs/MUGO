'''
previous extract step 
'''
from pyfaidx import Fasta
import pyfaidx
import kipoiseq
from kipoiseq import Interval
import time 
import numpy as np
import pandas as pd 
import sys 
import os 
import argparse
import pickle 

# new_column_names = ['chrom', 'pos', 'id', 'ref', 'alt','qual','filter', 'NaN', 'HG00096', 'HG00097', 'HG00099', 'HG00100', 'HG00101', 
#                     'HG00102', 'HG00103', 'HG00106', 'HG00108', 'HG00109', 'HG00110', 'HG00111', 'HG00112', 'HG00114', 'HG00116', 'HG00117',
#                      'HG00118', 'HG00119', 'HG00120', 'HG00121', 'HG00122', 'HG00123', 'HG00125', 'HG00126', 'HG00127', 'HG00128', 'HG00129',
#                   'HG00130', 'HG00131', 'HG00133', 'HG00136', 'HG00137', 'HG00138', 'HG00139', 'HG00141', 'HG00142', 'HG00143', 'HG00146',
#                    'HG00148', 'HG00149', 'HG00150', 'HG00151', 'HG00154', 'HG00155', 'HG00158', 'HG00159', 'HG00160', 'HG00171', 'HG00173',
#                     'HG00174', 'HG00176', 'HG00177', 'HG00178', 'HG00179', 'HG00180', 'HG00182', 'HG00183', 'HG00185', 'HG00186', 'HG00187', 
#                     'HG00188', 'HG00189', 'HG00231', 'HG00232', 'HG00233', 'HG00234', 'HG00235', 'HG00236', 'HG00238', 'HG00239', 'HG00240',
#                     'HG00242', 'HG00243', 'HG00244', 'HG00245', 'HG00246', 'HG00250', 'HG00251', 'HG00252', 'HG00253', 'HG00255', 'HG00256',
#                      'HG00257', 'HG00258', 'HG00259', 'HG00260', 'HG00261', 'HG00262', 'HG00263', 'HG00264', 'HG00265', 'HG00266', 'HG00267',
#                         'HG00268', 'HG00269', 'HG00271', 'HG00272', 'HG00273', 'HG00274', 'HG00275', 'HG00276', 'HG00277', 'HG00278', 'HG00280', 
#                      'HG00281', 'HG00282', 'HG00284', 'HG00285', 'HG00306', 'HG00309', 'HG00310', 'HG00311', 'HG00313', 'HG00315', 'HG00319', 
#                     'HG00320', 'HG00321', 'HG00323', 'HG00324', 'HG00325', 'HG00326', 'HG00327', 'HG00328', 'HG00329', 'HG00330', 'HG00331',
#                     'HG00332', 'HG00334', 'HG00335', 'HG00336', 'HG00337', 'HG00338', 'HG00339', 'HG00341', 'HG00342', 'HG00343', 'HG00344', 
#                     'HG00345', 'HG00346', 'HG00349', 'HG00350', 'HG00351', 'HG00353', 'HG00355', 'HG00356', 'HG00358', 'HG00360', 'HG00361', 
#                     'HG00362', 'HG00364', 'HG00366', 'HG00367', 'HG00369', 'HG00372', 'HG00373', 'HG00375', 'HG00376', 'HG00378', 'HG00381',
#                     'HG00382', 'HG00383', 'HG00384', 'HG01334', 'NA06984', 'NA06986', 'NA06989', 'NA06994', 'NA07037', 'NA07048', 'NA07051', 
#                     'NA07056', 'NA07347', 'NA07357', 'NA10847', 'NA10851', 'NA11829', 'NA11830', 'NA11831', 'NA11843', 'NA11892', 'NA11893', 
#                     'NA11894', 'NA11920', 'NA11930', 'NA11931', 'NA11992', 'NA11994', 'NA11995', 'NA12004', 'NA12006', 'NA12043', 'NA12044',
#                     'NA12045', 'NA12058', 'NA12144', 'NA12154', 'NA12155', 'NA12249', 'NA12272', 'NA12273', 'NA12275', 'NA12282', 'NA12283',
#                      'NA12286', 'NA12287', 'NA12340', 'NA12341', 'NA12342', 'NA12347', 'NA12348', 'NA12383', 'NA12399', 'NA12400', 'NA12413',
#                      'NA12489', 'NA12546', 'NA12716', 'NA12717', 'NA12718', 'NA12749', 'NA12750', 'NA12751', 'NA12761', 'NA12763', 'NA12775',
#                       'NA12777', 'NA12778', 'NA12812', 'NA12814', 'NA12815', 'NA12827', 'NA12829', 'NA12830', 'NA12842', 'NA12843', 'NA12872', 
#                       'NA12873', 'NA12874', 'NA12889', 'NA12890', 'NA18486', 'NA18489', 'NA18498', 'NA18499', 'NA18502', 'NA18505', 'NA18508', 
#                       'NA18510', 'NA18511', 'NA18517', 'NA18519', 'NA18520', 'NA18858', 'NA18861', 'NA18867', 'NA18868', 'NA18870', 'NA18873',
#                        'NA18907', 'NA18908', 'NA18909', 'NA18910', 'NA18912', 'NA18916', 'NA18917', 'NA18923', 'NA18933', 'NA18934', 'NA19093', 
#                        'NA19095', 'NA19096', 'NA19098', 'NA19099', 'NA19102', 'NA19107', 'NA19108', 'NA19113', 'NA19114', 'NA19116', 'NA19117', 
#                        'NA19118', 'NA19119', 'NA19121', 'NA19129', 'NA19130', 'NA19131', 'NA19137', 'NA19138', 'NA19146', 'NA19147', 'NA19149',
#                         'NA19152', 'NA19160', 'NA19171', 'NA19172', 'NA19175', 'NA19185', 'NA19189', 'NA19190', 'NA19197', 'NA19198', 'NA19200',
#                          'NA19204', 'NA19207', 'NA19209', 'NA19213', 'NA19222', 'NA19223', 'NA19225', 'NA19235', 'NA19236', 'NA19247', 'NA19248',
#                           'NA19256', 'NA19257', 'NA20502', 'NA20503', 'NA20504', 'NA20505', 'NA20506', 'NA20507', 'NA20508', 'NA20509', 'NA20510', 
#                           'NA20512', 'NA20513', 'NA20515', 'NA20516', 'NA20517', 'NA20518', 'NA20519', 'NA20520', 'NA20521', 'NA20524', 'NA20525', 
#                       'NA20527', 'NA20528', 'NA20529', 'NA20530', 'NA20531', 'NA20532', 'NA20534', 'NA20535', 'NA20536', 'NA20538', 'NA20539',
#                        'NA20540', 'NA20541', 'NA20542', 'NA20543', 'NA20544', 'NA20581', 'NA20582', 'NA20585', 'NA20586', 'NA20588', 'NA20589',
#                         'NA20752', 'NA20754', 'NA20756', 'NA20757', 'NA20758', 'NA20759', 'NA20760', 'NA20761', 'NA20765', 'NA20766', 'NA20768',
#                          'NA20769', 'NA20770', 'NA20771', 'NA20772', 'NA20773', 'NA20774', 'NA20778', 'NA20783', 'NA20785', 'NA20786', 'NA20787',
#                         'NA20790', 'NA20792', 'NA20795', 'NA20796', 'NA20797', 'NA20798', 'NA20799', 'NA20800', 'NA20801', 'NA20802', 'NA20803',
#                       'NA20804', 'NA20805', 'NA20806', 'NA20807', 'NA20808', 'NA20809', 'NA20810', 'NA20811', 'NA20812', 'NA20813', 'NA20814', 
#                         'NA20815', 'NA20819', 'NA20826', 'NA20828']

new_column_names = ['chrom', 'pos', 'id', 'ref', 'alt','qual','filter', 'NaN', 'HG00096', 'HG00097', 'HG00099', 'HG00100', 'HG00101', 'HG00102', 
                    'HG00103', 'HG00104', 'HG00106', 'HG00108', 'HG00109', 'HG00110', 'HG00111', 'HG00112', 'HG00114', 'HG00116', 'HG00117', 'HG00118',
                     'HG00119', 'HG00120', 'HG00121', 'HG00122', 'HG00123', 'HG00124', 'HG00125', 'HG00126', 'HG00127', 'HG00128', 'HG00129', 'HG00130',
                     'HG00131', 'HG00133', 'HG00134', 'HG00135', 'HG00136', 'HG00137', 'HG00138', 'HG00139', 'HG00141', 'HG00142', 'HG00143', 'HG00146',
                    'HG00148', 'HG00149', 'HG00150', 'HG00151', 'HG00152', 'HG00154', 'HG00155', 'HG00156', 'HG00158', 'HG00159', 'HG00160', 'HG00171', 
                    'HG00173', 'HG00174', 'HG00176', 'HG00177', 'HG00178', 'HG00179', 'HG00180', 'HG00182', 'HG00183', 'HG00185', 'HG00186', 'HG00187',
                      'HG00188', 'HG00189', 'HG00231', 'HG00232', 'HG00233', 'HG00234', 'HG00235', 'HG00236', 'HG00238', 'HG00239', 'HG00240', 'HG00242',
                     'HG00243', 'HG00244', 'HG00245', 'HG00246', 'HG00247', 'HG00249', 'HG00250', 'HG00251', 'HG00252', 'HG00253', 'HG00255', 'HG00256',
                    'HG00257', 'HG00258', 'HG00259', 'HG00260', 'HG00261', 'HG00262', 'HG00263', 'HG00264', 'HG00265', 'HG00266', 'HG00267', 'HG00268',
                    'HG00269', 'HG00271', 'HG00272', 'HG00273', 'HG00274', 'HG00275', 'HG00276', 'HG00277', 'HG00278', 'HG00280', 'HG00281', 'HG00282', 
                    'HG00284', 'HG00285', 'HG00306', 'HG00309', 'HG00310', 'HG00311', 'HG00312', 'HG00313', 'HG00315', 'HG00319', 'HG00320', 'HG00321', 
                    'HG00323', 'HG00324', 'HG00325', 'HG00326', 'HG00327', 'HG00328', 'HG00329', 'HG00330', 'HG00331', 'HG00332', 'HG00334', 'HG00335', 
                    'HG00336', 'HG00337', 'HG00338', 'HG00339', 'HG00341', 'HG00342', 'HG00343', 'HG00344', 'HG00345', 'HG00346', 'HG00349', 'HG00350', 
                    'HG00351', 'HG00353', 'HG00355', 'HG00356', 'HG00358', 'HG00359', 'HG00360', 'HG00361', 'HG00362', 'HG00364', 'HG00366', 'HG00367', 
                    'HG00369', 'HG00372', 'HG00373', 'HG00375', 'HG00376', 'HG00377', 'HG00378', 'HG00381', 'HG00382', 'HG00383', 'HG00384', 'HG01334', 
                    'NA06984', 'NA06986', 'NA06989', 'NA06994', 'NA07037', 'NA07048', 'NA07051', 'NA07056', 'NA07347', 'NA07357', 'NA10847', 'NA10851', 
                    'NA11829', 'NA11830', 'NA11831', 'NA11843', 'NA11892', 'NA11893', 'NA11894', 'NA11920', 'NA11930', 'NA11931', 'NA11992', 'NA11993', 
                    'NA11994', 'NA11995', 'NA12004', 'NA12006', 'NA12043', 'NA12044', 'NA12045', 'NA12058', 'NA12144', 'NA12154', 'NA12155', 'NA12249', 
                    'NA12272', 'NA12273', 'NA12275', 'NA12282', 'NA12283', 'NA12286', 'NA12287', 'NA12340', 'NA12341', 'NA12342', 'NA12347', 'NA12348', 
                    'NA12383', 'NA12399', 'NA12400', 'NA12413', 'NA12489', 'NA12546', 'NA12716', 'NA12717', 'NA12718', 'NA12749', 'NA12750', 'NA12751', 
                    'NA12761', 'NA12763', 'NA12775', 'NA12777', 'NA12778', 'NA12812', 'NA12814', 'NA12815', 'NA12827', 'NA12829', 'NA12830', 'NA12842', 
                    'NA12843', 'NA12872', 'NA12873', 'NA12874', 'NA12889', 'NA12890', 'NA18486', 'NA18487', 'NA18489', 'NA18498', 'NA18499', 'NA18502', 
                    'NA18505', 'NA18508', 'NA18510', 'NA18511', 'NA18517', 'NA18519', 'NA18520', 'NA18858', 'NA18861', 'NA18867', 'NA18868', 'NA18870', 
                    'NA18873', 'NA18907', 'NA18908', 'NA18909', 'NA18910', 'NA18912', 'NA18916', 'NA18917', 'NA18923', 'NA18933', 'NA18934', 'NA19093', 
                    'NA19095', 'NA19096', 'NA19098', 'NA19099', 'NA19102', 'NA19107', 'NA19108', 'NA19113', 'NA19114', 'NA19116', 'NA19117', 'NA19118', 
                    'NA19119', 'NA19121', 'NA19129', 'NA19130', 'NA19131', 'NA19137', 'NA19138', 'NA19146', 'NA19147', 'NA19149', 'NA19150', 'NA19152', 
                    'NA19160', 'NA19171', 'NA19172', 'NA19175', 'NA19185', 'NA19189', 'NA19190', 'NA19197', 'NA19198', 'NA19200', 'NA19204', 'NA19207', 
                    'NA19209', 'NA19213', 'NA19222', 'NA19223', 'NA19225', 'NA19235', 'NA19236', 'NA19247', 'NA19248', 'NA19256', 'NA19257', 'NA20502', 
                    'NA20503', 'NA20504', 'NA20505', 'NA20506', 'NA20507', 'NA20508', 'NA20509', 'NA20510', 'NA20512', 'NA20513', 'NA20515', 'NA20516', 
                    'NA20517', 'NA20518', 'NA20519', 'NA20520', 'NA20521', 'NA20524', 'NA20525', 'NA20527', 'NA20528', 'NA20529', 'NA20530', 'NA20531', 
                    'NA20532', 'NA20534', 'NA20535', 'NA20536', 'NA20537', 'NA20538', 'NA20539', 'NA20540', 'NA20541', 'NA20542', 'NA20543', 'NA20544', 
                    'NA20581', 'NA20582', 'NA20585', 'NA20586', 'NA20588', 'NA20589', 'NA20752', 'NA20754', 'NA20756', 'NA20757', 'NA20758', 'NA20759', 
                    'NA20760', 'NA20761', 'NA20765', 'NA20766', 'NA20768', 'NA20769', 'NA20770', 'NA20771', 'NA20772', 'NA20773', 'NA20774', 'NA20778', 
                    'NA20783', 'NA20785', 'NA20786', 'NA20787', 'NA20790', 'NA20792', 'NA20795', 'NA20796', 'NA20797', 'NA20798', 'NA20799', 'NA20800', 
                    'NA20801', 'NA20802', 'NA20803', 'NA20804', 'NA20805', 'NA20806', 'NA20807', 'NA20808', 'NA20809', 'NA20810', 'NA20811', 'NA20812', 
                    'NA20813', 'NA20814', 'NA20815', 'NA20816', 'NA20819', 'NA20826', 'NA20828'] # 421 in tatal, all for huang's 


# This will use chr1.fai if it exists, or create it if it doesn't
path_chr1 = '/projects/zhanglab/users/dongbo/personalized-expression-benchmark/data/ref_fasta/chr1.fa'
chr1 = Fasta(path_chr1)

# Access a specific region (e.g., from position 100,000 to 100,100)
sequence = chr1['chr1'][100000:100100].seq
print(sequence)

parser = argparse.ArgumentParser(description='Your program description')

# Add arguments
# parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
# parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
parser.add_argument('--gene_index', type=int, default=2704, # start from 1, 0 is titles . min is 1 
                    help='row of gene to extract in gene_3000.csv, 0 - 3014, will be read out from gene_3000.csv, which is genes has name')
# parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
# parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

args = parser.parse_args() # parse the arguments 


SEQ_LENGTH = 393_216

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


# Setup
consensus_dir = '/projects/zhanglab/users/dongbo/personalized-expression-benchmark/data/ref_fasta' # TODO path to consensus sequences (reference seq)
genes_file = '/projects/zhanglab/users/dongbo/gene_3000.csv'
SEQUENCE_LENGTH = 393216
INTERVAL = 114688

gene_df = pd.read_csv(genes_file, names=["gene_ID", "chr", "pos", "gene_name", "strand", "index_gene"])  # pos, i.e. tss here. 
print("## Starting predictions ##")

print("predicting gene: ", gene_df.loc[args.gene_index, "gene_name"])  # gene name.

gene_name = gene_df.loc[args.gene_index, "gene_name"]  # gene name. 

chr = gene_df.loc[args.gene_index, "chr"]  # 1, 2, 3... 
TSS = gene_df.loc[args.gene_index, "pos"] # transcription start site. 
print("gene is on chr: ",chr, " TSS: ",TSS) 

start = int(TSS) - SEQUENCE_LENGTH // 2
end = int(TSS) + SEQUENCE_LENGTH // 2 - 1
print("start: ",start, " end: ",end) 

file = f"{consensus_dir}/chr"+ str(chr) +".fa"   # './personalized-expression-benchmark/data/ref_fasta/chr1.fa'
fasta_extractor = FastaStringExtractor(file)  # extractor for reference. 

# Extract the sequence 
path_allindi = '/projects/zhanglab/users/dongbo/462_dataset/423_individual_list.pkl' 
with open(path_allindi, 'rb') as f:
    all_individuals = pickle.load(f)

print("total rows of all individual file after filter: ",len(all_individuals)) # 421
print(all_individuals) # debug


# import subprocess
# import os 
# path = f'/projects/zhanglab/users/dongbo/462_dataset/filtered_vcf/{gene_name}.csv' # for all 405 indis. 
# # Run the touch command

# if not os.path.exists(path):
#     subprocess.run(['touch',path])  # touch the csv file to save filtered vcf first.

path_csv = f'/projects/zhanglab/users/dongbo/462_dataset/filtered_vcf_huangdataset/{gene_name}.csv' # for all 405 indis. 
genetype_df = pd.read_csv(path_csv,error_bad_lines=False, header=None) # read the csv file. 
genetype_df.columns = new_column_names
print(genetype_df) # debug 


indis =  ['HG00096', 'HG00097', 'HG00099', 'HG00100', 'HG00101', 'HG00102', 
                    'HG00103', 'HG00104', 'HG00106', 'HG00108', 'HG00109', 'HG00110', 'HG00111', 'HG00112', 'HG00114', 'HG00116', 'HG00117', 'HG00118',
                     'HG00119', 'HG00120', 'HG00121', 'HG00122', 'HG00123', 'HG00124', 'HG00125', 'HG00126', 'HG00127', 'HG00128', 'HG00129', 'HG00130',
                     'HG00131', 'HG00133', 'HG00134', 'HG00135', 'HG00136', 'HG00137', 'HG00138', 'HG00139', 'HG00141', 'HG00142', 'HG00143', 'HG00146',
                    'HG00148', 'HG00149', 'HG00150', 'HG00151', 'HG00152', 'HG00154', 'HG00155', 'HG00156', 'HG00158', 'HG00159', 'HG00160', 'HG00171', 
                    'HG00173', 'HG00174', 'HG00176', 'HG00177', 'HG00178', 'HG00179', 'HG00180', 'HG00182', 'HG00183', 'HG00185', 'HG00186', 'HG00187',
                      'HG00188', 'HG00189', 'HG00231', 'HG00232', 'HG00233', 'HG00234', 'HG00235', 'HG00236', 'HG00238', 'HG00239', 'HG00240', 'HG00242',
                     'HG00243', 'HG00244', 'HG00245', 'HG00246', 'HG00247', 'HG00249', 'HG00250', 'HG00251', 'HG00252', 'HG00253', 'HG00255', 'HG00256',
                    'HG00257', 'HG00258', 'HG00259', 'HG00260', 'HG00261', 'HG00262', 'HG00263', 'HG00264', 'HG00265', 'HG00266', 'HG00267', 'HG00268',
                    'HG00269', 'HG00271', 'HG00272', 'HG00273', 'HG00274', 'HG00275', 'HG00276', 'HG00277', 'HG00278', 'HG00280', 'HG00281', 'HG00282', 
                    'HG00284', 'HG00285', 'HG00306', 'HG00309', 'HG00310', 'HG00311', 'HG00312', 'HG00313', 'HG00315', 'HG00319', 'HG00320', 'HG00321', 
                    'HG00323', 'HG00324', 'HG00325', 'HG00326', 'HG00327', 'HG00328', 'HG00329', 'HG00330', 'HG00331', 'HG00332', 'HG00334', 'HG00335', 
                    'HG00336', 'HG00337', 'HG00338', 'HG00339', 'HG00341', 'HG00342', 'HG00343', 'HG00344', 'HG00345', 'HG00346', 'HG00349', 'HG00350', 
                    'HG00351', 'HG00353', 'HG00355', 'HG00356', 'HG00358', 'HG00359', 'HG00360', 'HG00361', 'HG00362', 'HG00364', 'HG00366', 'HG00367', 
                    'HG00369', 'HG00372', 'HG00373', 'HG00375', 'HG00376', 'HG00377', 'HG00378', 'HG00381', 'HG00382', 'HG00383', 'HG00384', 'HG01334', 
                    'NA06984', 'NA06986', 'NA06989', 'NA06994', 'NA07037', 'NA07048', 'NA07051', 'NA07056', 'NA07347', 'NA07357', 'NA10847', 'NA10851', 
                    'NA11829', 'NA11830', 'NA11831', 'NA11843', 'NA11892', 'NA11893', 'NA11894', 'NA11920', 'NA11930', 'NA11931', 'NA11992', 'NA11993', 
                    'NA11994', 'NA11995', 'NA12004', 'NA12006', 'NA12043', 'NA12044', 'NA12045', 'NA12058', 'NA12144', 'NA12154', 'NA12155', 'NA12249', 
                    'NA12272', 'NA12273', 'NA12275', 'NA12282', 'NA12283', 'NA12286', 'NA12287', 'NA12340', 'NA12341', 'NA12342', 'NA12347', 'NA12348', 
                    'NA12383', 'NA12399', 'NA12400', 'NA12413', 'NA12489', 'NA12546', 'NA12716', 'NA12717', 'NA12718', 'NA12749', 'NA12750', 'NA12751', 
                    'NA12761', 'NA12763', 'NA12775', 'NA12777', 'NA12778', 'NA12812', 'NA12814', 'NA12815', 'NA12827', 'NA12829', 'NA12830', 'NA12842', 
                    'NA12843', 'NA12872', 'NA12873', 'NA12874', 'NA12889', 'NA12890', 'NA18486', 'NA18487', 'NA18489', 'NA18498', 'NA18499', 'NA18502', 
                    'NA18505', 'NA18508', 'NA18510', 'NA18511', 'NA18517', 'NA18519', 'NA18520', 'NA18858', 'NA18861', 'NA18867', 'NA18868', 'NA18870', 
                    'NA18873', 'NA18907', 'NA18908', 'NA18909', 'NA18910', 'NA18912', 'NA18916', 'NA18917', 'NA18923', 'NA18933', 'NA18934', 'NA19093', 
                    'NA19095', 'NA19096', 'NA19098', 'NA19099', 'NA19102', 'NA19107', 'NA19108', 'NA19113', 'NA19114', 'NA19116', 'NA19117', 'NA19118', 
                    'NA19119', 'NA19121', 'NA19129', 'NA19130', 'NA19131', 'NA19137', 'NA19138', 'NA19146', 'NA19147', 'NA19149', 'NA19150', 'NA19152', 
                    'NA19160', 'NA19171', 'NA19172', 'NA19175', 'NA19185', 'NA19189', 'NA19190', 'NA19197', 'NA19198', 'NA19200', 'NA19204', 'NA19207', 
                    'NA19209', 'NA19213', 'NA19222', 'NA19223', 'NA19225', 'NA19235', 'NA19236', 'NA19247', 'NA19248', 'NA19256', 'NA19257', 'NA20502', 
                    'NA20503', 'NA20504', 'NA20505', 'NA20506', 'NA20507', 'NA20508', 'NA20509', 'NA20510', 'NA20512', 'NA20513', 'NA20515', 'NA20516', 
                    'NA20517', 'NA20518', 'NA20519', 'NA20520', 'NA20521', 'NA20524', 'NA20525', 'NA20527', 'NA20528', 'NA20529', 'NA20530', 'NA20531', 
                    'NA20532', 'NA20534', 'NA20535', 'NA20536', 'NA20537', 'NA20538', 'NA20539', 'NA20540', 'NA20541', 'NA20542', 'NA20543', 'NA20544', 
                    'NA20581', 'NA20582', 'NA20585', 'NA20586', 'NA20588', 'NA20589', 'NA20752', 'NA20754', 'NA20756', 'NA20757', 'NA20758', 'NA20759', 
                    'NA20760', 'NA20761', 'NA20765', 'NA20766', 'NA20768', 'NA20769', 'NA20770', 'NA20771', 'NA20772', 'NA20773', 'NA20774', 'NA20778', 
                    'NA20783', 'NA20785', 'NA20786', 'NA20787', 'NA20790', 'NA20792', 'NA20795', 'NA20796', 'NA20797', 'NA20798', 'NA20799', 'NA20800', 
                    'NA20801', 'NA20802', 'NA20803', 'NA20804', 'NA20805', 'NA20806', 'NA20807', 'NA20808', 'NA20809', 'NA20810', 'NA20811', 'NA20812', 
                    'NA20813', 'NA20814', 'NA20815', 'NA20816', 'NA20819', 'NA20826', 'NA20828'] # 421 in total

# filter to save only len of ref and alt is 1. 
genetype_df = genetype_df[genetype_df.apply(lambda row: len(row['alt']) <= 1 and len(row['ref']) <= 1, axis=1)]

# fileter MAF 
MAF = 0.05 
NUM_ALLELE = 2*len(indis) # 842

# Create a function to calculate MAF count
def calculate_MAF_count(row):
    MAF_count = 0
    for col_name in indis:
        data = row[col_name]
        if data=='0|1' or data=='1|0':
            MAF_count += 1 
        elif data=='1|1': 
            MAF_count += 2
    return MAF_count

# Calculate MAF count for each row
genetype_df['MAF_count'] = genetype_df[indis].apply(calculate_MAF_count, axis=1)

# Calculate MAF ratio
genetype_df['MAF_ratio'] = genetype_df['MAF_count'] / NUM_ALLELE

# Filter rows based on MAF ratio
genetype_df = genetype_df[(genetype_df['MAF_ratio'] >= MAF) & (genetype_df['MAF_ratio'] <= 1-MAF)]

# Drop the temporary columns
genetype_df.drop(['MAF_count', 'MAF_ratio'], axis=1, inplace=True)
genetype_df = genetype_df.reset_index(drop=True) # reset the index. 
print(genetype_df)  # debug


'''
plug snp into ref seq below, and save in csv 
'''

df = pd.DataFrame(columns=["geneId", "chr", "tss", "gene_name", "individual", "mater_seq", "pater_seq"])  # save the sequence of individual. mater is first, pater is second.
target_interval = kipoiseq.Interval(f'chr{chr}', start, end)
sequence_letter = fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)) # 393216 
# Convert the string to a list
sequence_list = list(sequence_letter) # ['A','T', ...]  do not change length of list during revising, otherwise will index wrong 

for i in range(len(indis)): # iterate all individuals 405.
    current_indi = indis[i] # get individual name. HG00319 
    # print("0 done")
    print("now extract individual: ",current_indi, " index: ",i) 

    paternal_sequence_list = sequence_list.copy()  # copy the list, otherwise will change the original list. ['A','T', ...], first number of 0|0
    maternal_sequence_list = sequence_list.copy()  # copy the list, otherwise will change the original list. ['A','T', ...], second number of 0|0 

    # print("2 done") # bottleneck between 2 to 3, 

    # print("before list: ",sequence_list[805:820]) # debug 
    # iterate over each SNP, and replace letter in sequence_letter. 
    for index, row in genetype_df.iterrows(): #  iterate over each row. about 500 SNPs in total 
    # for j in range(len(filter_snps)): # 10000+ SNPs, iterate SNP on gene. 
        # print("1 done") # check bottleneck 

        pos = row['pos'] - start - 1  # pos of the SNP, 500 SNP in total. 
        ref = row['ref']  # ref is the ref allele. 
        alt = row['alt']  # alt is the alt allele. 
        # snp_idx = bim_filtered.iloc[j]['idx'] 

        snp = row[current_indi]   # 0|1, for each indi and each SNP. 

        # print("at pos: ",pos, "ref: ",ref, "alt: ",alt, "snp: ",snp, "before:", sequence_list[pos]) # debug SNP mainly 0|0, which is ref. 

        # print("at pos: ",pos, "ref: ",ref, "alt: ",alt, "snp: ",snp, "before:", sequence_list[pos]) # debug 
        if snp == '0|0': # 0 is ref, 1 is alt. 
            # print("at pos: ",pos, "ref: ",ref, "alt: ",alt, "snp: ",snp, "before:", sequence_list[pos]) # debug 
            paternal_sequence_list[pos] = ref # ref is 'A'
            maternal_sequence_list[pos] = ref # ref is 'A' 
            # print("after paternal:", paternal_sequence_list[pos]) # debug 
            # print("after maternal:", maternal_sequence_list[pos]) # debug 

        elif snp == '0|1':  # paternal is first, maternal is second. 
            paternal_sequence_list[pos] = ref
            maternal_sequence_list[pos] = alt
            # print("after paternal:", paternal_sequence_list[pos]) # debug 
            # print("after maternal:", maternal_sequence_list[pos]) # debug 

        elif snp == '1|0':  # paternal is first, maternal is second.
            paternal_sequence_list[pos] = alt
            maternal_sequence_list[pos] = ref
            # print("after paternal:", paternal_sequence_list[pos]) # debug 
            # print("after maternal:", maternal_sequence_list[pos]) # debug 

        
        elif snp == '1|1':  # 1 is alt.
            paternal_sequence_list[pos] = alt
            maternal_sequence_list[pos] = alt
            # print("after paternal:", paternal_sequence_list[pos]) # debug 
            # print("after maternal:", maternal_sequence_list[pos]) # debug 

    paternal_sequence_letter = ''.join(paternal_sequence_list)
    maternal_sequence_letter = ''.join(maternal_sequence_list)
        # Append the row to the DataFrame
    df = df.append({  # name is gene name, add patient id later. 
        "geneId": gene_df.loc[args.gene_index, "gene_ID"],
        "chr": gene_df.loc[args.gene_index, "chr"],
        "tss": gene_df.loc[args.gene_index, "pos"],
        "gene_name": gene_df.loc[args.gene_index, "gene_name"],
        "strand": gene_df.loc[args.gene_index, "strand"],
        "paternal_sequence": paternal_sequence_letter,
        "maternal_sequence": maternal_sequence_letter,
        "individual": current_indi   # current indi name. 
    }, ignore_index=True)


# Write the DataFrame to a CSV file
gene_name = gene_df.loc[args.gene_index, "gene_name"] # SCYL3 
# path = '/projects/zhanglab/users/dongbo/462_dataset/geneseq_all_gene_Huangdataset/' + gene_name +'_index' + str(args.gene_index) + '_individual_seq.csv' 
# path = '/projects/zhanglab/users/dongbo/462_dataset/geneseq_all_gene_Huangdataset/' + gene_name +'_index' + str(args.gene_index) + '_individual_seq_1and2.csv' 
path = '/projects/zhanglab/users/dongbo/462_dataset/geneseq_all_gene_Huangdataset_allSNP/' + gene_name +'_index' + str(args.gene_index) + '_individual_seq.csv' 
df.to_csv(path, index=False)




