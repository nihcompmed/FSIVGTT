import numpy as np
from chung_cpep_secretion import parameters

data_file = open('all_data_ffa.txt','r')
data = data_file.readlines()
print(len(data))
print(data[0])
subject_list = list(set([x.split()[0] for x in data if x[0]!='s']))
valid_subjects = []
subject_data = dict()
for subj in subject_list:
    subject_data[subj] = [y.split()[1:] for y in data if y.split()[0]==subj]
    if len(subject_data[subj])>1:
        valid_subjects.append(subj)
num_data = dict()
for subj in valid_subjects:
    num_data[subj] = np.vstack(\
        [np.array([float(y[-5]),float(y[-4]),float(y[-3]),float(y[-2]),float(y[-1])]) for y in subject_data[subj] if y[-1]!='.'])
    #time
#    print(subj,num_data[subj].shape)
data_file.close()

secretion_file = open('Chung_minimal_model.txt','w')
secretion_file.write('subject'+'\t'+'cx'+\
                     '\t'+'Ibx'+\
                     '\t'+'l0'+\
                     '\t'+'l1'+\
                     '\t'+'x2'+\
                     '\t'+'cf'+\
                     '\t'+'xcl'+\
                     '\t'+'SG'+\
                     '\t'+'SI'+\
                     '\t'+'Gb'+\
                     '\n')
for subject in valid_subjects:
    subj0 = num_data[subject]
    print(subject)
    par = parameters(subj0)
secretion_file.close()
#print(secr(subj0))
