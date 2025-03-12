#!/usr/bin/env python3
import csv

# 该文件的作用是将csv格式转换为标准的tum位姿格式

input_file = './dataset/SubT_MRS/SubT_MRS_Urban_Challenge_UGV2/poses/ground_truth_path.csv'
output_file = './dataset/SubT_MRS/SubT_MRS_Urban_Challenge_UGV2/poses/gt_poses_tum.txt'

with open(input_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header
    with open(output_file, 'w') as outfile:
        for row in reader:
            nsec, x, y, z, qx, qy, qz, qw = map(float, row)
            sec = nsec * 1e-9  # Convert nanoseconds to seconds
            output_line = f"{sec} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
            outfile.write(output_line)

print("Conversion completed, file saved as", output_file)
