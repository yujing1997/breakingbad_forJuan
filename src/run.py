"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import os
from pathlib import Path
import json
from glob import glob
import SimpleITK as sitk

from inference import Segmentator


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # Read the input
    ct_path  = load_image_file_as_array(
        location=INPUT_PATH / "images/ct",
    )
    if os.path.exists(INPUT_PATH / "ehr.json"):
        input_electronic_health_record = load_json_file(
            location=INPUT_PATH / "ehr.json",
        )
    pt_path = load_image_file_as_array(
        location=INPUT_PATH / "images/pet",
    )
    
    ct = sitk.ReadImage(ct_path)
    pet = sitk.ReadImage(pt_path)
    segmentator = Segmentator(folds=0, dataset_name="Dataset002_pet_ct_noMD", tile_step_size=0.5)
    seg, metadata = segmentator.predict(image_ct=ct, image_pet=pet, preprocess=True, return_logits=False)
    
    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/tumor-lymph-node-segmentation",
        array=seg,
    )
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    return input_files[0]


def write_array_as_image_file(*, location, array, filename="output.mha"):
    location.mkdir(parents=True, exist_ok=True)

    if isinstance(array, sitk.Image):
        img = array                       # already sitk
    else:                                 # assume NumPy
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]              # drop batch dim if present
        img = sitk.GetImageFromArray(array)

    sitk.WriteImage(img, str(location / filename), useCompression=True)



def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())