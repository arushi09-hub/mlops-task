#To open the file.
programfile = open('/root/mlops/model.py','r')
#Read the code
code = programfile.read()

if 'keras' or 'tensorflow' in code:
    print('NEURAL NETWORK CODE')
else:
    print('NOT A NEURAL NETWORK')
