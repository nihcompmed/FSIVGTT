import numpy as np
from chung_cpep_secretion import secretion as secr

data_file = open('all_data.txt','r')
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
        [np.array([float(y[-4]),float(y[-1])]) for y in subject_data[subj] if y[-1]!='.'])
#    print(subj,num_data[subj].shape)
data_file.close()

subj_file = open('data.txt','r')
subj_parameters = subj_file.readlines()
headings = subj_parameters[0].split()
print(headings)

headings = headings[1:]
subject_parameters = dict()
subject_list = list(set([x.split()[0] for x in data if x[0]!='s']))
for subj in valid_subjects:
    z_temp = [y.split()[1:] for y in subj_parameters if y.split()[0]==subj]
#    print(len(z_temp[0]),len(headings))
    subject_parameters[subj] = dict()
    for i,heading in enumerate(headings):
        subject_parameters[subj][heading] = z_temp[0][i]
subj_file.close()

secretion_file = open('Chung_secretion_5_1e6.txt','w')
secretion_file.write('subject'+'\t'+'time_min'+\
                     '\t'+'Secretion rate'+\
                     '\n')
for subject in valid_subjects:
    subj0 = (subject_parameters[subject],num_data[subject])
    print(subject)
    times,secr_subj,var_subj = secr(subj0)
#    del_subj = np.copy(del_subj[:,0]).reshape(var_subj.shape)
    num_rows = secr_subj.shape[0]
    index=0
    secretion_file.write(subject+'\t'+str(0.0)+\
                         '\t'+str(abs(secr_subj[index]))+\
                         '\t 0.0\n')
    for index in range(1,num_rows):
        secretion_file.write(subject+'\t'+str(times[index-1])+\
                             '\t'+str(secr_subj[index])+\
                             '\t'+str(var_subj[index-1])+'\n')
secretion_file.close()
#print(secr(subj0))
