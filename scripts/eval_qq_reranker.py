import csv
import sys
from operator import itemgetter

if (len(sys.argv) < 3):
  print "Usage:", sys.argv[0], "labels_file score_file high (default: higher score is better)"


labelfile=sys.argv[1]
scorefile=sys.argv[2]
higher=sys.argv[3]
rev=False
if higher == "high":
  rev=True

groundtruth=[]
sentences=[]
scores=[]
unique_sentences=set()
with open(labelfile,'r') as f:
    reader=csv.reader(f,delimiter='\t')
    for a,b,c in reader:
        groundtruth.append(a)
        sentences.append(b)
        unique_sentences.add(b)

with open(scorefile) as f:
  for score in f:
    scores.append(float(score.strip()))

# print(unique_sentences)

cool = [list(x) for x in sorted(zip(groundtruth, sentences, scores),
    key=itemgetter(2), reverse=rev)]

used_sentences=set()
counter = 0
true_count = 0
wrong_count = 0
for truth,sent,score in cool:
  if sent in used_sentences:
    pass
  else:
    counter += 1
    if truth == "1":
      true_count += 1
    else:
      wrong_count += 1
    used_sentences.add(sent)

print counter, true_count, wrong_count
recall = (true_count*1.0/counter)
precision = (true_count*1.0/(true_count + wrong_count))
print "Precision/Recall", precision, recall

