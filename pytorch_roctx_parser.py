import os
import sys
import argparse

import operator
import collections

class OperatorSummary():
    def __init__(self, name, count, kernel_time, percentage):
        self.kernel_name = name
        self.kernel_count = count
        self.kernel_time = kernel_time
        self.kernel_percentage = percentage

        self.all_ops = []

    def __str__(self):
        return ("Operator summary Name: {}, Count: {}, Time: {}, Percentage: {}".format(self.kernel_name, self.kernel_count, self.kernel_time, self.kernel_percentage))
        
def parse_json_file(json_file):
    fs = open(json_file, 'r')
    lines = fs.readlines()
    fs.close()

    useful_lines = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("seq =" in line and "name" in line and "dur" in line):
            useful_lines.append(line)

    #print (len(useful_lines))
    return useful_lines

def parse_useful_lines(useful_lines):

    kernel_names = []
    kernel_times = []

    for j in range(len(useful_lines)):
        line = useful_lines[j]
        #print (line)
        values = line.split(",")
        for i in range(len(values)):
            if "name" in values[i]:
                kernel_name = values[i].split('"name"')[1].replace('"','')[1:]
                #print (kernel_name)
                kernel_names.append(kernel_name)
            if "dur" in values[i]:
                #print (values[i])
                duration = values[i].split('"dur":')[1].replace('"', '')
                #print (duration)
                kernel_times.append(float(duration)/1000000)

    return kernel_names, kernel_times

def get_top_kernel_summary(kernel_names, kernel_times):
    kernel_summary = {}
    for i in range(len(kernel_names)):
        name = kernel_names[i]
        ktime = kernel_times[i]
        if name not in kernel_summary:
            kernel_summary[name] = ktime
        else:
            kernel_summary[name] = kernel_summary[name] + ktime

    #print(len(kernel_summary))
    #print(kernel_summary.items())
    #sorted_dict = {k : v for k, v in sorted(kernel_summary.items(), key=lamdba item: item[1])}
    sorted_dict = sorted(kernel_summary.items(), key=operator.itemgetter(1), reverse=True)
    
    #print (sorted_dict)
    for j in range(len(sorted_dict)):
        print (sorted_dict[j])
        
def dump_call_trace_and_times(kernel_names, kernel_times):
    
    #assert len(kernel_names) == len(kernel_times)
    
    fs = open("net_pytorch_kernel_call_trace.csv", 'w')
    fs.write('sep=|')
    fs.write('\n')
    for j in range(len(kernel_names)):
        fs.write(kernel_names[j] + ',' + str(kernel_times[j]))
        fs.write('\n')
    fs.close()

    print ("INFO: Dumped out all pytorch kernel call trace into net_pytorch_kernel_call_trace.csv")

def get_total_kernel_time(kernel_times):
    total_kernel_time = 0
    for j in range(len(kernel_times)):
        total_kernel_time += kernel_times[j]
    print ("INFO: Total kernel time is : {}".format(total_kernel_time))
    return total_kernel_time

def main():
    json_file = os.path.abspath(args.json_file)
    
    useful_lines = parse_json_file(json_file)
    kernel_names, kernel_times = parse_useful_lines(useful_lines)

    dump_call_trace_and_times(kernel_names, kernel_times)
    total_kernel_time = get_total_kernel_time(kernel_times)
    get_top_kernel_summary(kernel_names, kernel_times)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str, required=True,help="JSON File after rocprof + rocpd")
    
    args = parser.parse_args()
    main()
