import os
import sys
# os.system("gcc -o ravi find_systolic_peaks.c findpeaks.c string1.c rtGetInf.c rt_nonfinite.c sort1.c sortIdx.c rtGetNaN.c eml_setop.c")
# os.system("gcc -o neu neural.c rtGetInf.c rt_nonfinite.c sort1.c sortIdx.c rtGetNaN.c eml_setop.c")

code = sys.argv[1]
print(f"Argument passed: {code}")
command = f"gcc -o peak_run {code}.c"
os.system(command)
os.system("peak_run.exe")