import time
import os
import logging
logger = logging.getLogger(__name__)
from typing import Union, List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import SimpleITK as sitk
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.paths import nnUNet_results
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Add the paths so constants, preprocessing, and postprocessing utilities can be found
# regardless of working directory
src_dir = os.path.dirname(os.path.abspath(__file__))

# Add the src directory and its subdirectories to Python path
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'constants'))
sys.path.insert(0, os.path.join(src_dir, 'preprocessing'))
sys.path.insert(0, os.path.join(src_dir, 'postprocessing'))

from constants.labels import ids
from preprocessing.utils import resample_pet_to_ct
from preprocessing.cropping import crop_z_axis, crop_below_brain
from postprocessing.utils import pad, edit_label_metadata


class Segmentator():
    def __init__(self, folds:Union[List[int], Tuple[int], int]=(0, 1, 2, 3, 4), 
                 dataset_name:str="Dataset002_pet_ct_noMD", 
                 tile_step_size:float=0.9, config:str="3d_fullres_bs8", 
                 checkpoint:str="best"):
        
        assert os.environ.get('nnUNet_results') is not None, "Please set the environment variable nnUNet_results to the path of your nnUNet results folder."
        # instantiate the nnUNetPredictor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        ##  nnUNet validation args
        # tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        #                             perform_everything_on_device=True, device=self.device, verbose=False,
        #                             verbose_preprocessing=False, allow_tqdm=False

        self.predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring= True,
            perform_everything_on_device=True,
            device=device,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True,
        )
        # initializes the network architecture, loads the checkpoint
        if isinstance(folds, int):
            folds = [folds]
        assert checkpoint in ["best", "final"], "checkpoint must be either 'best' or 'final'"
        if checkpoint == "best":
            checkpoint_name = "checkpoint_best.pth"
        else:
            checkpoint_name = "checkpoint_final.pth"
        self.predictor.initialize_from_trained_model_folder(
            join(
                nnUNet_results, f"{dataset_name}/nnUNetTrainer__nnUNetPlans__{config}"
            ),
            checkpoint_name=checkpoint_name,
            use_folds=folds,
        )

        super().__init__()

    def predict(self, *, 
        image_ct:sitk.Image, 
        image_pet:sitk.Image,
        margin_mm_side_brain:float = 50.0,
        margin_mm_below_brain:float = 10.0,
        preprocess:bool=True,
        return_logits:bool=False
    ) -> sitk.Image:

        if ('mem_optimized' in os.environ.keys()) and (os.environ['mem_optimized'].lower() in ('true', '1', 't')):
            print("RUNNING INFERENCE WITH MEMORY OPTIMIZED")

        assert image_ct is not None or image_pet is not None, "pass either ct or mri or both"

        # knowing how much time the inference takes
        t0_prediction = time.time()

        torch.cuda.empty_cache()

        if image_ct is not None:
            image_ct = sitk.Cast(image_ct, sitk.sitkFloat32)
        if image_pet is not None:
            image_pet = sitk.Cast(image_pet, sitk.sitkFloat32)

        # We save target metadata
        target_spacing = image_ct.GetSpacing()
        target_direction = image_ct.GetDirection()
        target_origin = image_ct.GetOrigin()
        target_size = image_ct.GetSize()
       
        if preprocess: 
            
            image_pet = resample_pet_to_ct(image_pet, image_ct)

            ct_cropped_brain, pet_cropped_brain = crop_below_brain(
                ct=image_ct, pet=image_pet, mask=None, margin_mm=margin_mm_side_brain, save=False)
            
            image_ct, image_pet = crop_z_axis(
                ct_cropped_brain, pet_cropped_brain, mask=None, margin_mm=margin_mm_below_brain, save=False)
    

        if image_ct is not None:
            spacing = image_ct.GetSpacing()
            direction = image_ct.GetDirection()
            origin = image_ct.GetOrigin()
            shape = image_ct.GetSize()
        else:
            spacing = image_pet.GetSpacing()
            direction = image_pet.GetDirection()
            origin = image_pet.GetOrigin()
            shape = image_pet.GetSize()

        images = []
        spacings = []
        spacings.append(spacing)
        spacings_for_nnunet = []

        npy_ct = sitk.GetArrayFromImage(image_ct)
        npy_ct = npy_ct[None]

        npy_pet = sitk.GetArrayFromImage(image_pet)
        npy_pet = npy_pet[None]
                
        spacings_for_nnunet.append(list(spacings[-1])[::-1])

        images.append(npy_ct)
        images.append(npy_pet)

        spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        dict = {
            "sitk_stuff": {
                "spacing": spacings[0],
            },
            "spacing": spacings_for_nnunet[0],
        }

        # print(len(images), images[0].shape, images[1].shape, images[-1].shape)
        images = np.vstack(images).astype(np.float32)

        # time for preprocessing
        t1_preprocessing = time.time()
       
        del npy_ct, npy_pet
            
        assert len(self.predictor.dataset_json["channel_names"]) == len(images), (
            f"Your model needs {len(self.predictor.dataset_json['channel_names'])} inputs but you give it {len(images)}."
        )
        outputs = self.predictor.predict_single_npy_array(
            input_image=images, 
            image_properties=dict,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=return_logits,
            return_times=True,
        )
        if return_logits:
            output_seg, probabilities, t1_inference, t1_resampling = outputs
            print("logits shape ", probabilities.shape)
            print("output_seg shape ", output_seg.shape)
            print("UNIQUE VALUES ", np.unique(probabilities)[:10], probabilities.shape)
            print("max value in probabilities ", np.max(probabilities))
            print("UNIQUE VALUES ", np.unique(output_seg)[:10], output_seg.shape)
        else:
            output_seg, t1_inference, t1_resampling = outputs

        # output_seg = self.predictor.predict_sliding_window_return_logits(
        #     torch.from_numpy(images)
        # ).cpu().numpy().astype(float)
        
        output_seg = sitk.GetImageFromArray(output_seg)
        # Here we set to metadata of the cropped array
        output_seg.SetDirection(direction)
        output_seg.SetOrigin(origin)
        output_seg.SetSpacing(spacing)

        if return_logits:
            logits = sitk.GetImageFromArray(probabilities[1])
            logits.SetDirection(direction)
            logits.SetOrigin(origin)
            logits.SetSpacing(spacing)
        # Here we pad with the original metadata as target
        if preprocess:
            output_seg = pad(
                volume_to_pad=output_seg,
                spacing=target_spacing,
                size=target_size,
                origin=target_origin,
                direction=target_direction,
            )
            if return_logits:
                logits = pad(
                    volume_to_pad=logits,
                    spacing=target_spacing,
                    size=target_size,
                    origin=target_origin,
                    direction=target_direction,
                    interpolator=sitk.sitkBSpline
                )
        assert np.sum(sitk.GetArrayFromImage(output_seg)) >0
        # output should be a sitk image with the same size, spacing, origin and direction as the original input image_ct
        output_seg = edit_label_metadata(sitk.Cast(output_seg, sitk.sitkUInt8), ids)

        # time for decombining left right and padding
        t1_postprocessing = time.time()

        metadata = {
            "t0_prediction": t0_prediction,
            "t1_preprocessing": t1_preprocessing,
            "t1_inference": t1_inference,
            "t1_resampling": t1_resampling,
            "t1_postprocessing": t1_postprocessing,
            "shape": shape,
            "spacing": spacing,
        }   
        if return_logits:
            return output_seg, logits, metadata
        return output_seg, metadata


if __name__ == "__main__":
    
    # Set nnUNet_results environment variable if not already set
    if os.environ.get('nnUNet_results') is None:
        os.environ['nnUNet_results'] = "/Data/Yujing/HECKTOR2025/Hecktor2025/nnUNet_results_submission"
        print(f"Set nnUNet_results to: {os.environ['nnUNet_results']}")

    # ct = sitk.ReadImage("/media/yujing/800129L/Head_and_Neck/HECKTOR_Challenge/HECKTOR 2025 Task 2 Training/Task 2/CHUM-001/CHUM-001__CT.nii.gz")
    # pet = sitk.ReadImage("/media/yujing/800129L/Head_and_Neck/HECKTOR_Challenge/HECKTOR 2025 Task 2 Training/Task 2/CHUM-001/CHUM-001__PT.nii.gz")

    ct = sitk.ReadImage("/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/ct/CHUM-001.mha")
    pet = sitk.ReadImage("/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/pet/CHUM-001.mha")

    segmentator = Segmentator(folds=0, dataset_name="Dataset002_pet_ct_noMD", tile_step_size=0.5)
    seg, logits, metadata = segmentator.predict(image_ct=ct, image_pet=pet, preprocess=True, return_logits=True)
    sitk.WriteImage(seg, "/Data/Yujing/HECKTOR2025/Hecktor2025/output/pred.mha", True)
    sitk.WriteImage(seg, "/Data/Yujing/HECKTOR2025/Hecktor2025/output/pred.nii.gz", True)
    sitk.WriteImage(logits, "/Data/Yujing/HECKTOR2025/Hecktor2025/output/logits.nii.gz", True)
    sitk.WriteImage(logits, "/Data/Yujing/HECKTOR2025/Hecktor2025/output/logits.seg.nrrd", True)

# running an inference 
# cd /Data/Yujing/HECKTOR2025/Hecktor2025/src && export nnUNet_results="/Data/Yujing/HECKTOR2025/Hecktor2025/nnUNet_results_submission" && python inference.py
