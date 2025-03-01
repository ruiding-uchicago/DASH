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
def sum_all(object_list):
    sum_result=0
    for strings in object_list:
        b=int(strings)
        sum_result+=b
    return sum_result

m= MPRester(my_key)
def generate_ads_on_opted_slab(opted_slab,adsorbate):
    asf_each_init=AdsorbateSiteFinder(opted_slab)
    ads_structs = asf_each_init.generate_adsorption_structures(adsorbate, repeat=[1,1,1])
    return ads_structs
def compute_and_find_nearest_structures_number(adsorbate_struct_list,adsobate_atom_number,species_number,adsobate_atom_O_number,exclude_element_list=['O','F','I','Br']):
    min_distance_dict={}
    ads_slab_dict={}
    min_distance_dict_EX={}
    ads_slab_dict_EX={}
    weighted_dict={}
    for i in range(0,len(adsorbate_struct_list)):
        target_ads_struct=adsorbate_struct_list[i]
        ####
        distance_dict={}
        distance_dict_EX={}
        for j in range(0,len(adsorbate_struct_list[i])-species_number*adsobate_atom_number):
            each_atom=adsorbate_struct_list[i][j]
            if each_atom.species_string not in exclude_element_list:
                
                each_distance_to_the_adsorbed_O_atom=each_atom.distance_from_point(adsorbate_struct_list[i][-adsobate_atom_O_number].coords)
                distance_dict[each_atom]=each_distance_to_the_adsorbed_O_atom
                min_distance=min(distance_dict.values())
            else:
                each_distance_to_the_adsorbed_O_atom_EX=each_atom.distance_from_point(adsorbate_struct_list[i][-adsobate_atom_O_number].coords)
                distance_dict_EX[each_atom]=each_distance_to_the_adsorbed_O_atom_EX
                min_distance_EX=min(distance_dict_EX.values())
                

        the_nearest_atom=min(distance_dict, key=distance_dict.get)
        the_nearest_atom_EX=min(distance_dict_EX, key=distance_dict_EX.get)
        ###对于某一个子结构，先找出举例吸附O最近的那个原子
        ###对第一个结构，min_distance_dict直接存放这个结构里的最近原子和最近原子对应的最近距离
        ###第二个结构开始，如果最近原子是同一原子，对比是否更小，如果更小，那么直接替代
        try:
            if min_distance_dict[the_nearest_atom]>min_distance:
                min_distance_dict[the_nearest_atom]=min_distance
        except:
            min_distance_dict[the_nearest_atom]=min_distance
        try:
            if min_distance_dict_EX[the_nearest_atom_EX]>min_distance_EX:
                min_distance_dict_EX[the_nearest_atom_EX]=min_distance_EX
        except:
            min_distance_dict_EX[the_nearest_atom_EX]=min_distance_EX
        #####
        ads_slab_dict[(i,str(the_nearest_atom.specie))]=min_distance
        ads_slab_dict_EX[(i,str(the_nearest_atom_EX.specie))]=min_distance_EX
        weighted_dict[(i,str(the_nearest_atom.specie))]=min_distance/min_distance_EX
        print('the'+str(i)+'th struct computed finished')
    print(ads_slab_dict) 
    print(ads_slab_dict_EX)
    print(weighted_dict)
    
    serial_dict=find_the_serial_number_in_the_list(weighted_dict)
    print(serial_dict)
    return_list=[]
    for each_serial in serial_dict:
        return_list.append(serial_dict[each_serial][0])
    return return_list

def compute_and_find_nearest_structures_number_specific(adsorbate_struct_list,adsobate_atom_number,species_number,adsobate_atom_O_number,target_element,exclude_element_list=['O','F','I','Br']):
    min_distance_dict={}
    ads_slab_dict={}
    min_distance_dict_EX={}
    ads_slab_dict_EX={}
    weighted_dict={}
    for i in range(0,len(adsorbate_struct_list)):
        target_ads_struct=adsorbate_struct_list[i]
        ####
        distance_dict={}
        distance_dict_EX={}
        for j in range(0,len(adsorbate_struct_list[i])-species_number*adsobate_atom_number):
            each_atom=adsorbate_struct_list[i][j]
            if each_atom.species_string not in exclude_element_list:
                
                each_distance_to_the_adsorbed_O_atom=each_atom.distance_from_point(adsorbate_struct_list[i][-adsobate_atom_O_number].coords)
                distance_dict[each_atom]=each_distance_to_the_adsorbed_O_atom
                min_distance=min(distance_dict.values())
            else:
                each_distance_to_the_adsorbed_O_atom_EX=each_atom.distance_from_point(adsorbate_struct_list[i][-adsobate_atom_O_number].coords)
                distance_dict_EX[each_atom]=each_distance_to_the_adsorbed_O_atom_EX
                min_distance_EX=min(distance_dict_EX.values())
                

        the_nearest_atom=min(distance_dict, key=distance_dict.get)
        the_nearest_atom_EX=min(distance_dict_EX, key=distance_dict_EX.get)
        ###对于某一个子结构，先找出举例吸附O最近的那个原子
        ###对第一个结构，min_distance_dict直接存放这个结构里的最近原子和最近原子对应的最近距离
        ###第二个结构开始，如果最近原子是同一原子，对比是否更小，如果更小，那么直接替代
        try:
            if min_distance_dict[the_nearest_atom]>min_distance:
                min_distance_dict[the_nearest_atom]=min_distance
        except:
            min_distance_dict[the_nearest_atom]=min_distance
        try:
            if min_distance_dict_EX[the_nearest_atom_EX]>min_distance_EX:
                min_distance_dict_EX[the_nearest_atom_EX]=min_distance_EX
        except:
            min_distance_dict_EX[the_nearest_atom_EX]=min_distance_EX
        #####
        ads_slab_dict[(i,str(the_nearest_atom.specie))]=min_distance
        ads_slab_dict_EX[(i,str(the_nearest_atom_EX.specie))]=min_distance_EX
        weighted_dict[(i,str(the_nearest_atom.specie))]=min_distance/min_distance_EX
        print('the'+str(i)+'th struct computed finished')
    print(ads_slab_dict) 
    print(ads_slab_dict_EX)
    print(weighted_dict)
    
    serial_dict=find_the_serial_number_in_the_list(weighted_dict)
    print(serial_dict)
    return serial_dict[target_element][0]

def find_the_serial_number_in_the_list(ads_slab_dict):
    output_dict={}
    for key in ads_slab_dict:
        serial_number=key[0]
        element_type=re.findall(r"\D+",key[1])[0]
#         print(element_type)
        distance=ads_slab_dict[key]
#         print(output_dict)
        try:        
#             print(output_dict[element_type])
            if distance<output_dict[element_type][1]:
#                 print("yes")
                output_dict[element_type]=(serial_number,distance)
        except:
            output_dict[element_type]=(serial_number,distance)
    return output_dict
def mkdir(path): 
    folder = os.path.exists(path) 
    if not folder:
        os.makedirs(path)
        print ("---  new folder...  ---")
        print ("---  OK  ---") 
    else:
        print ("---  There is this folder!  ---")
    
def read_struct_and_make_poscar(path,adsorbate,adsorbate_name,adsobate_atom_number,species_number,adsobate_atom_O_number):
    dir_list=os.listdir(path)
    print(dir_list)
    for dir_each in dir_list:
        try:
            dir_each=path+dir_each
            Read_Struct=Structure.from_file(dir_each+'/CONTCAR')
            adsorbate_struct_list=generate_ads_on_opted_slab(Read_Struct,adsorbate)
            adsorbate_struct_list_copy=adsorbate_struct_list.copy()
            for each in adsorbate_struct_list:
                each.make_supercell([[1,0,0],
                          [0,1,0],
                          [0,0,1]])
            write_series=compute_and_find_nearest_structures_number(adsorbate_struct_list,adsobate_atom_number,species_number,adsobate_atom_O_number)
            print(write_series)
            for i in write_series:
                mkdir(dir_each+'/ADS_'+adsorbate_name+'/ADS_'+str(i))
                open(dir_each+'/ADS_'+adsorbate_name+'/ADS_'+str(i)+'/POSCAR' , 'w').write(str(Poscar(adsorbate_struct_list_copy[i])))
                mpr=MPRelaxSet(adsorbate_struct_list_copy[i])
                mpr.write_input(output_dir=dir_each+'/ADS_'+adsorbate_name+'/ADS_'+str(i))

        except:
            print('file not exist')

def read_struct_and_make_poscar_specific_element(path,adsorbate,adsorbate_name,adsobate_atom_number,species_number,adsobate_atom_O_number,target_element):
    dir_list=os.listdir(path)
    print(dir_list)
    for dir_each in dir_list:
        try:
            dir_each=path+dir_each
            Read_Struct=Structure.from_file(dir_each+'/CONTCAR')
            adsorbate_struct_list=generate_ads_on_opted_slab(Read_Struct,adsorbate)
            adsorbate_struct_list_copy=adsorbate_struct_list.copy()
            for each in adsorbate_struct_list:
                each.make_supercell([[1,0,0],
                          [0,1,0],
                          [0,0,1]])
            write_series=compute_and_find_nearest_structures_number_specific(adsorbate_struct_list,adsobate_atom_number,species_number,adsobate_atom_O_number,target_element)
            print(write_series)
            mkdir(dir_each+'/ADS_'+adsorbate_name+'/ADS_'+str(write_series))
            open(dir_each+'/ADS_'+adsorbate_name+'/ADS_'+str(write_series)+'/POSCAR' , 'w').write(str(Poscar(adsorbate_struct_list_copy[write_series])))
            mpr=MPRelaxSet(adsorbate_struct_list_copy[write_series])
            mpr.write_input(output_dir=dir_each+'/ADS_'+adsorbate_name+'/ADS_'+str(write_series))

        except Exception as e:
            print(e)

path = r'./'
adsorbate_O =  Molecule("O", [[0, 0, 0]])
adsorbate_OH =  Molecule("OH", [[0, 0, 0], [-0.793, 0.384, 0.422]])
adsorbate_OOH = Molecule("OOH", [[0, 0, 0], [-1.067, -0.403, 0.796], [-0.696, -0.272, 1.706]])
read_struct_and_make_poscar_specific_element(path,adsorbate_O,'O',1,1,1,'Ru')
read_struct_and_make_poscar_specific_element(path,adsorbate_OH,'OH',2,1,2,'Ru')
read_struct_and_make_poscar_specific_element(path,adsorbate_OOH,'OOH',3,1,3,'Ru')
