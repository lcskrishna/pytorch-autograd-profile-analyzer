import os
import sys
import argparse

def read_log_file(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    useful_lines = []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("train phase" in line and ("(100.00% done)" in line)):
            useful_lines.append(line)

    return useful_lines

def get_loss_values(useful_lines):
    fs = open('loss_values.csv', 'w')
    iters = 0
    for j in range(len(useful_lines)):
        line = useful_lines[j]
        split_values = line.split(',')
        loss_values = split_values[1]
        loss = loss_values.split(':')[1]
        print_line = str(iters) + ',' + str(loss)
        fs.write(print_line)
        fs.write('\n')
        iters = iters + 1
    fs.close()        

def main():
    log_file = os.path.abspath(args.log_file)
    useful_lines = read_log_file(log_file)
    get_loss_values(useful_lines)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True, help="Log file after training video classification.")
    
    args = parser.parse_args()
    main()
