import os
import sys
import argparse

#MI100 names.
#categories = {
#               'BLAS' : ['aten::mm', 'aten::matmul', 'MmBackward', 'aten::bmm', 'BmmBackward0', 'aten::addmm', 'AddmmBackward',],
#               'Eltwise' : ['aten::sum', 'aten::add', 'aten::mul_', 'aten::mul', 'aten::add_', 'aten::div', 'MulBackward0', 'DivBackward0', 'aten::exp', 'aten::max', 
#                            'TanhBackward',],
#               'Distributed' : ['_ReduceFromModelParallelRegion', '_CopyToModelParallelRegionBackward', '_VocabParallelCrossEntropy'],
#               'Normalization' : ['aten::norm', 'FusedLayerNormAffineFunctionBackward', 'FusedLayerNormAffineFunction'],
#               'Embedding' : ['aten::index_put_', 'EmbeddingBackward', 'aten::index_select', 'aten::index', 'IndexPutBackward',],
#               'Softmax/GELU/Dropout' : ['FusedDropoutBackward', 'aten::dropout', 'SoftmaxBackward', 'aten::softmax', 'aten::gelu', 'LogSoftmaxBackward', 'log_softmax' ],
#               'MemoryRelatedCalls' : ['aten::copy_', 'aten::to', 'aten::cat', 'aten::item', 'torch::autograd::AccumulateGrad', 'aten::contiguous', 'aten::clone',
#                                       'SplitBackward', 'aten::reshape', 'aten::zeros'],
#               'LossFunctions' : ['NllLossBackward', 'aten::nll_loss'],
#               'Other' : ['aten::nonzero', 'aten::arange', 'aten::lt', 'aten::ge', 'aten::rsub'], 
#             }

# V100 names.
categories = {
               'BLAS' : ['mm', 'matmul', 'MmBackward', 'bmm', 'BmmBackward0', 'addmm', 'AddmmBackward',],
               'Eltwise' : ['sum', 'add', 'mul_', 'mul', 'add_', 'div', 'MulBackward0', 'DivBackward0', 'exp', 'max',
                            'TanhBackward',],
               'Distributed' : ['_ReduceFromModelParallelRegion', '_CopyToModelParallelRegionBackward', '_VocabParallelCrossEntropy'],
               'Normalization' : ['norm', 'FusedLayerNormAffineFunctionBackward', 'FusedLayerNormAffineFunction'],
               'Embedding' : ['index_put_', 'EmbeddingBackward', 'index_select', 'index', 'IndexPutBackward',],
               'Softmax/GELU/Dropout' : ['FusedDropoutBackward', 'dropout', 'SoftmaxBackward', 'softmax', 'gelu', 'LogSoftmaxBackward', 'log_softmax' ],
               'MemoryRelatedCalls' : ['copy_', 'to', 'cat', 'item', 'torch::autograd::AccumulateGrad', 'contiguous', 'clone',
                                       'SplitBackward', 'reshape', 'zeros', 'normal_','cat_out',],
               'LossFunctions' : ['NllLossBackward', 'nll_loss'],
               'Other' : ['nonzero', 'arange', 'lt', 'ge', 'rsub', 'empty_strided', 'empty_like',],
             }

def read_csv_file(csv_file):
    fs = open(csv_file, 'r')
    lines = fs.readlines()
    fs.close()

    operators = []
    for j in range(1, len(lines)):
        line = lines[j].rstrip()
        operators.append(line)

    return operators

def list_kernel_summary(operators):
    category_keys = categories.keys()
    for key in category_keys:
        values = categories[key]
        total_duration = 0.0
        for j in range(len(operators)):
            op = operators[j]
            vals = op.split('|')
            name = vals[0]
            if (vals[0] in values):
                total_duration += float(vals[1])

        print ("Category : {}, TotalTime: {}".format(key, total_duration))
        #print (key)
        #print (total_duration)
                
def main():
    csv_file = os.path.abspath(args.csv_file)
    operators = read_csv_file(csv_file)
    list_kernel_summary(operators)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str, required=True, help="Operator CSV File")
    
    args = parser.parse_args()
    main()
