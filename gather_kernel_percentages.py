import os
import sys
import argparse

import re

categories = {
                'BLAS' : ['Cijk'],
                'MIOpen' : ['miopen', 'MIOpen', 'Im2d2Col','gridwise','OpTensor', 'SubTensorOp', 'Col2Im2d', 'transpose_', 'SubSample', ],
                'Eltwise' : ['elementwise_kernel',],
                'Reduce' : ['reduce_kernel',],
                'ROI' : ['RoI'],
                'Index' : ['index'], 
             }

def read_csv_file(csv_file):
    fs = open(csv_file, 'r')
    lines = fs.readlines()
    fs.close()
    
    kernel_names = []
    kernel_times = []
    kernel_percentages = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("sep=|" in line):
            continue
        elif ("Name" in line):
            continue
        else:
            values = line.split("|")
            kernel_names.append(values[0])
            kernel_times.append(float(values[2])/1000000)
            kernel_percentages.append(float(values[4]))

    return kernel_names, kernel_times, kernel_percentages

def process_categories(kernel_names, kernel_times, kernel_percentages):
    
    category_keys = categories.keys()
    for key in category_keys:
        values = categories[key]
        total_time = 0.0
        total_percentage = 0.0
        for j in range(len(kernel_names)):
            name = kernel_names[j]
            for i in range(len(values)):
                if (key == "Eltwise"):
                    if re.search(values[i], name) and not re.search("index", name):
                        total_time += kernel_times[j]
                        total_percentage += kernel_percentages[j]
                else:
                    if re.search(values[i], name):
                        total_time += kernel_times[j]
                        total_percentage += kernel_percentages[j]
            
        print ("Category: {}, TotalTime: {}, TotalPercentage: {}".format(key, total_time, total_percentage))                   

def main():
    csv_file = os.path.abspath(args.csv_file)
    kernel_names, kernel_times, kernel_percentages = read_csv_file(csv_file)
    process_categories(kernel_names, kernel_times, kernel_percentages) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", type=str, required=True, help="CSV File generated out of rocmProfileData.")
    
    args = parser.parse_args()
    main() 
