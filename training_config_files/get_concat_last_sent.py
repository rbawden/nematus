import os
import sys

def find_last(lst, elm):
  gen = (len(lst) - 1 - i for i, v in enumerate(reversed(lst)) if v == elm)
  return next(gen, None)

def get_last():                                           

    several_concats = 0
    no_concat = 0

    for line in sys.stdin:
        tabline = line.strip().split()
        if "<CONCAT>" in tabline:
            if tabline.count("<CONCAT>") > 1:
                os.sys.stderr.write("more than one concats...\n")
                several_concats+=1
            last_concat = find_last(tabline, "<CONCAT>")
            os.sys.stdout.write(" ".join(tabline[last_concat+1:])+"\n")
        else:
            os.sys.stderr.write("no concat...\n")
            no_concat+=1
            os.sys.stdout.write(line.strip()+"\n")

    os.sys.stderr.write("Several concats: "+str(several_concats)+", No concat : "+str(no_concat)+"\\
n")
                                                                     
get_last()
