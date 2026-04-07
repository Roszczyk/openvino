import os
from pathlib import Path

CONTEXT = Path(__file__).parent
file_list = [f for f in os.listdir(CONTEXT) if os.path.isfile(CONTEXT / f)]

zeKernelCreate_all_count = 0
zeKernelDestroy_all_count = 0
zeModuleCreate_all_count = 0
zeModuleDestroy_all_count = 0

for file_name in file_list:
    with open(CONTEXT / file_name, "r") as f:
        file_content = f.read()

    zeKernelCreate_count = file_content.count("zeKernelCreate")
    zeKernelDestroy_count = file_content.count("zeKernelDestroy")
    zeModuleCreate_count = file_content.count("zeModuleCreate")
    zeModuleDestroy_count = file_content.count("zeModuleDestroy")

    zeKernelCreate_all_count += zeKernelCreate_count
    zeKernelDestroy_all_count += zeKernelDestroy_count
    zeModuleCreate_all_count += zeModuleCreate_count
    zeModuleDestroy_all_count += zeModuleDestroy_count

    print(f"""
===============================================
FILE: {file_name}
zeKernelCreate : {zeKernelCreate_count}
zeKernelDestroy : {zeKernelDestroy_count}
zeModuleCreate : {zeModuleCreate_count}
zeModuleDestroy : {zeModuleDestroy_count}
DELTAS: {zeKernelCreate_count - zeKernelDestroy_count} || {zeModuleCreate_count - zeModuleDestroy_count}
===============================================
""")
    
print(f"""
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
ALL FILES SUMMARY:
zeKernelCreate : {zeKernelCreate_all_count}
zeKernelDestroy : {zeKernelDestroy_all_count}
zeModuleCreate : {zeModuleCreate_all_count}
zeModuleDestroy : {zeModuleDestroy_all_count}
DELTAS: {zeKernelCreate_all_count - zeKernelDestroy_all_count} || {zeModuleCreate_all_count - zeModuleDestroy_all_count}   
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
""")