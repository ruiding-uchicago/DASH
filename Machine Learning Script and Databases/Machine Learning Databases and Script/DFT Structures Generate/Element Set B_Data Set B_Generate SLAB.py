#encoding:utf-8
my_key='BJU2qDd4gpCPO0rf'
from pymatgen.core import Structure, Lattice
from pymatgen.core import Molecule
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matplotlib import pyplot as plt
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.inputs import PotcarSingle
from numpy import array
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.ext.matproj import MPRester
import re
import numpy as np
def sum_all(object_list):
    sum_result=0
    for strings in object_list:
        b=int(strings)
        sum_result+=b
    return sum_result
import random
m= MPRester(my_key)

def sum_all_float(object_list):
    sum_result=0
    for b in object_list:
        sum_result=sum_result+b
    return sum_result
def subset(alist, idxs):

    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list
def generate_float(group_num,min_ratio,max_ratio,seed):
    random.seed(seed)
    original_list=[]
    for i in range(0,group_num):
        original_list.append(random.uniform(min_ratio,max_ratio))
    if sum_all_float(original_list)<1:
        return original_list
    else:
        return_list=[]
        for each in original_list:
            return_list.append(0.8*each/sum_all_float(original_list))
        return return_list
def sum_each_by_each(target_list):
    return_list=[]
    for i in range(0,len(target_list)):
        current_sum=sum_all_float(target_list[:i+1])
        return_list.append(current_sum)
    return return_list
def multiple_list(target_list,constant):
    return_list=[]
    for i in range(0,len(target_list)):
        return_list.append(target_list[i]*constant)
    return return_list
def multiple_list_int(target_list,constant):
    return_list=[]
    for i in range(0,len(target_list)):
        return_list.append(int(target_list[i]*constant))
    return return_list
def split_list_uniform(alist, group_num=4, shuffle=True, retain_left=False):


    index = list(range(len(alist))) # 保留下标

    # 是否打乱列表
    if shuffle: 
        random.shuffle(index) 
    
    
    elem_num = len(alist) // group_num # 每一个子列表所含有的元素数量
    sub_lists = {}
    
    # 取出每一个子列表所包含的元素，存入字典中
    for idx in range(group_num):
        start, end = idx*elem_num, (idx+1)*elem_num
        sub_lists['set'+str(idx)] = subset(alist, index[start:end])
    
    # 是否将最后剩余的元素作为单独的一组
    if retain_left and group_num * elem_num != len(index): # 列表元素数量未能整除子列表数，需要将最后那一部分元素单独作为新的列表
        sub_lists['set'+str(idx+1)] = subset(alist, index[end:])
    
    return sub_lists
def split_list_random_ratio(alist, min_ratio,max_ratio,seed, group_num=5,shuffle=True):

    random.seed(seed)
    index = list(range(len(alist))) # 保留下标

    # 是否打乱列表
    if shuffle: 
        random.shuffle(index) 
    print(index)
    list_random_1=generate_float(group_num-1, min_ratio,max_ratio,seed)
    list_random_2=sum_each_by_each(list_random_1)
    print(list_random_2)
    sub_lists = {}
    
    # 取出每一个子列表所包含的元素，存入字典中
    new_index_list=multiple_list_int(list_random_2,len(alist))
    print('节点index是',new_index_list)
    
    # 是否将最后剩余的元素作为单独的一组
    for i in range(0,len(new_index_list)):
        print(i)
        if i==0:
            current_index_list=index[0:new_index_list[i]]
        else:
            current_index_list=index[new_index_list[i-1]:new_index_list[i]]
        sub_lists['set'+str(i)]=(subset(alist,current_index_list))
    sub_lists['set'+str(len(new_index_list))]=(subset(alist,index[new_index_list[-1]:]))
    return sub_lists
def random_substitute(selected_slab,element_list,substitue_target_element,num_other_element,min_substitue_ratio,max_substitue_ratio,seed):
    print(seed)
    random.seed(seed)
    atom_dict=selected_slab.as_dict()
    target_series_list=[]
    for i in range(0,selected_slab.num_sites):
        if selected_slab[i].species_string==substitue_target_element:
            target_series_list.append(i)
    
    substitue_element_list=random.sample(element_list,num_other_element)
    substitue_element_list.append(substitue_target_element)
    #random.shuffle(substitue_element_list)
    print(substitue_element_list)
    
    series_shuffle_result=split_list_random_ratio(target_series_list,min_substitue_ratio,max_substitue_ratio,seed, group_num=num_other_element+1, shuffle=True)
    print(series_shuffle_result)
    for j in range(0,num_other_element+1):
        subset_name='set'+str(j)
        position=series_shuffle_result[subset_name]
        for k in position:
            atom_dict['sites'][k]['label']=substitue_element_list[j]
            atom_dict['sites'][k]['species'][0]['element']=substitue_element_list[j]


    slab_after_substitute=selected_slab.from_dict(atom_dict)
    
    return slab_after_substitute

def random_substitute_extra(selected_slab,element_list,extra_element_list,substitue_target_element,num_other_element,min_substitue_ratio,max_substitue_ratio,seed):
    print(seed)
    random.seed(seed)
    atom_dict=selected_slab.as_dict()
    target_series_list=[]
    for i in range(0,selected_slab.num_sites):
        if selected_slab[i].species_string==substitue_target_element:
            target_series_list.append(i)
    
    # Choose one element from each list or both from extra_element_list
    if random.random() < 0.5:  # 50% chance to choose one from each
        substitue_element_list = [random.choice(element_list), random.choice(extra_element_list)]
    else:  # 50% chance to choose both from extra_element_list
        substitue_element_list = random.sample(extra_element_list, 2)
    
    substitue_element_list.append(substitue_target_element)
    print(substitue_element_list)
    
    series_shuffle_result=split_list_random_ratio(target_series_list,min_substitue_ratio,max_substitue_ratio,seed, group_num=num_other_element+1, shuffle=True)
    print(series_shuffle_result)
    for j in range(0,num_other_element+1):
        subset_name='set'+str(j)
        position=series_shuffle_result[subset_name]
        for k in position:
            atom_dict['sites'][k]['label']=substitue_element_list[j]
            atom_dict['sites'][k]['species'][0]['element']=substitue_element_list[j]

    slab_after_substitute=selected_slab.from_dict(atom_dict)
    
    return slab_after_substitute
    
import os 
def mkdir(path): 
    folder = os.path.exists(path) 
    if not folder:
        os.makedirs(path)
        print ("---  new folder...  ---")
        print ("---  OK  ---") 
    else:
        print ("---  There is this folder!  ---")

element_list=['K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Rb','Sr','Y','Zr','Nb','Mo','Tc','Rh','Cs','Ba','Os','Re','Ir']
extra_element_list=['Ga', 'In', 'Sn', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Hg', 'Tl', 'Pb', 'Bi', 'Li', 'Al']
RuO2=m.get_structure_by_material_id('mp-825')
struct = SpacegroupAnalyzer(RuO2).get_conventional_standard_structure()
struct=reorient_z(struct)
slabs = generate_all_slabs(struct, 1,12.0, 15.0, center_slab=True)
m_index=(1,1,0)
slab_dict = {slab.miller_index:slab for slab in slabs}                   
RuO2_110=slab_dict[m_index]
RuO2_110.make_supercell([[2,0,0],
                          [0,2,0],
                          [0,0,1]])


for i in range (0,1500):
    new_slab=random_substitute_extra(RuO2_110,element_list,extra_element_list,'Ru',2,0.05,0.35,seed=i)
    path='./the_'+str(i)+'_th_structure/'
    mkdir('the_'+str(i)+'_th_structure')
    try:
        open(path+'POSCAR','w').write(str(Poscar(new_slab,sort_structure=True)))
    except Exception as e:
        print('e')
    mpr=MPRelaxSet(new_slab)
    mpr.write_input(output_dir=path)

