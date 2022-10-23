import argparse

def merge(opt):
    """Merge Train/Validation img_path-gt file because we use AIHUB data as Train data """
    f= open(opt.path1,'r')
    f_merge = open(opt.merge_path,'w')
    lines=f.readlines()
    for i in range(len(lines)):
            f_merge.write(lines[i])
    f.close()
    f= open(opt.path2,'r')
    lines=f.readlines()
    for i in range(len(lines)):
            f_merge.write(lines[i])
    f.close()
    f_merge.close()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', default="data/gt_train.txt", help='1st gt file ')
    parser.add_argument('--path2', default="data/gt_valid.txt", help='2nd gt file')
    parser.add_argument('--merge_path', default="data/gt_merge.txt", help='output file path')

    opt = parser.parse_args()
    merge(opt)