import os
import sys
import argparse
import json
import operator

#{'name': 'aten::_local_scalar_dense', 'ph': 'f', 'ts': 46425873.665, 'tid': 7, 'pid': 'CUDA functions', 'id': 54635, 'cat': 'cpu_to_cuda', 'args': {}}

def get_all_cuda_operators(json_data):
    cuda_objs = []
    for j in range(len(json_data)):
        json_obj = json_data[j]
        if (json_obj['pid'] == 'CUDA functions' and json_obj['ph'] == 'X'):
            cuda_objs.append(json_obj)

    return cuda_objs

def parse_json_file(json_file):
    with open(json_file) as profile_json:
        json_data = json.load(profile_json)
    print ("INFO: JSON object loaded succesfully.")
    
    return json_data

def get_count_of_each_operator(cuda_objs):
    
    prof_obj = {}
    for j in range(len(cuda_objs)):
        cuda_obj = cuda_objs[j]
        if (cuda_obj['name'] not in prof_obj):
            name = cuda_obj['name']
            dur = cuda_obj['dur']
            prof_obj[name] = dur
        else:
            name = cuda_obj['name']
            dur = cuda_obj['dur']
            original_dur = prof_obj[name]
            prof_obj[name] = dur + original_dur

    sorted_prof_obj = sorted(prof_obj.items(), key=operator.itemgetter(1), reverse=True)
    #for obj in sorted_prof_obj:
    #    print (obj) 
    return sorted_prof_obj

def print_top_ten_kernels(operators):
    print ("Name, Time(s)")
    for i in range(10):
        op = operators[i]
        name = op[0]
        total_dur = float(op[1])/1000000.0 
        print (name + "," + str(total_dur))

def dump_operators(operators):
    fs = open('operators.csv', 'w')
    fs.write('sep=|')
    fs.write('\n')
    for i in range(len(operators)):
        op = operators[i]
        name = op[0]
        total_dur = float(op[1])/1000000.0
        op_str = name + "|" + str(total_dur)
        fs.write(op_str)
        fs.write('\n')
    fs.close()
    print ("INFO: Dumped the operators in CSV file.")
    

def main():
    json_data = parse_json_file(os.path.abspath(args.json_file))
    cuda_objs = get_all_cuda_operators(json_data)
    operators = get_count_of_each_operator(cuda_objs)
    dump_operators(operators)
    if args.top:
        print_top_ten_kernels(operators)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-file', type=str, required=True, help="JSON File for loading.")
    parser.add_argument('--top', action='store_true', required=False, help="Print Top 10 kernels")
    parser.add_argument('--kernel-summary', action='store_true', required=False, help="Generate a Kernel summary based on category")

    args = parser.parse_args()
    main()
