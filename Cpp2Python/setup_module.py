import os, sys, getopt

input_dir = os.getcwd()
out_dir = 'D:\\python\\DLLs'

opts, args = getopt.getopt(sys.argv[1:],'hi:o:m:')
for op, value in opts:
    if op == '-o':
        out_dir =  value
    elif op == '-m':
        module_name = value

project_name = input_dir.split('\\')[-1];
src_dll = input_dir + '\\x64\\Release\\%s.dll'%project_name
src_py = input_dir + '\\%s\\%s.py'%(project_name, module_name)

dst_pyd = out_dir + '\\_%s.pyd'%module_name
dst_py = out_dir + "\\%s.py"%module_name

os.system('copy %s %s'%(src_dll, dst_pyd))
os.system('copy %s %s'%(src_py, dst_py))

print('copy %s => %s'%(src_dll, dst_pyd))
print('copy %s => %s'%(src_py, dst_py))