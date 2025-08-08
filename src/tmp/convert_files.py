### To test the docker you can simply create
import SimpleITK as sitk
# pet = sitk.ReadImage("/Data/Seb/Hecktor2025/Task_1/HMR-030/HMR-030__PT.nii.gz")
# sitk.WriteImage(pet, "/home/sebq/HECKTOR2025/input/images/pet/HMR-030.mha", True)
# ct = sitk.ReadImage("/Data/Seb/Hecktor2025/Task_1/HMR-030/HMR-030__CT.nii.gz")
# sitk.WriteImage(ct, "/home/sebq/HECKTOR2025/input/images/ct/HMR-030.mha", True)

parent_folder = "/media/yujing/Seagate/HECKTOR2025/Task_2"

pet = sitk.ReadImage("/media/yujing/Seagate/HECKTOR2025/Task_2/CHUM-001/CHUM-001__PT.nii.gz")
sitk.WriteImage(pet, "/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/pet/CHUM-001.mha", True)
ct = sitk.ReadImage("/media/yujing/Seagate/HECKTOR2025/Task_2/CHUM-001/CHUM-001__CT.nii.gz")
sitk.WriteImage(ct, "/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/ct/CHUM-001.mha", True)