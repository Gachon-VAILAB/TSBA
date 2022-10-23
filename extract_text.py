#-*- coding: utf-8 -*-
from jamo import h2j,j2hcj
import argparse

def extract_text(opt):

    """text included from imagepath-label lists"""

    f= open(opt.input_path,'r')
    output = open(opt.output_path,'w')
    lines = f.readlines()
    for line in lines:
        data = line.split('\t')
        jamo = h2j(data[1])
        jam= j2hcj(jamo)
        """ find image_path - text that include jaeum in findings """
        findings = ["ㄲ", "ㄸ",
                    "ㅃ", "ㅆ" ,"ㅉ","ㄳ", "ㄵ", "ㄶ", "ㄺ", "ㄻ", "ㄼ", "ㄽ","ㄾ", "ㄿ", "ㅀ", "ㅄ"]
        if any(finding in jam for finding in findings):
            output.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="data/gt_merge.txt", help='input text file path to extract')
    parser.add_argument('--output_path', default="data/gt_jaeum.txt", help='extracted text file location')
    opt = parser.parse_args()
    extract_text(opt)


