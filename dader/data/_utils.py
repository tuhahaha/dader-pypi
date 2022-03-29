import csv
import pandas as pd

def read_csv(input_file, quotechar='"'):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
      reader = csv.reader(f,quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
      reader = csv.reader(f,delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

def norm(s):
    s = s.replace(","," ").replace("\'","").replace("\"","")
    return s
