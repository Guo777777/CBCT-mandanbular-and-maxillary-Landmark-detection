# legacy/

Experimental and one-off scripts from the original code base, kept **verbatim** (hard-coded
`/home/user16/...` paths included) for provenance.  Nothing in the released pipeline imports
them; the maintained versions live in `cbct_landmark/` and `scripts/`.

| file | what it was |
|---|---|
| `json_origin_explain.py` | prints origin / spacing / size of one case and converts its `*.mrk.json` landmarks to voxel indices (`(pos - origin) / spacing`) |
| `json_process.py` | first version of the crop + heat-map + evaluation script; everything but the size statistics is commented out. Superseded by `scripts/3_crop_tooth_roi.py` and `scripts/6_evaluate.py` |
| `json_process_test1.py` | dumps GT and predicted landmark voxel coordinates of the test set to `landmark1.csv` |
| `json_process_based_on_nibabel.py` | unfinished attempt to read the NRRD header with `pynrrd` |
| `image_process.py` | earlier ROI crop that assumed a 128^3 coarse mask (the released stage 1 uses 72^3) and wrote `crop_tooth_region*` folders |
| `zoom_in_tooth_image.py` | nearest-neighbour up-sampling of the 72^3 mask to the original grid (visual QA) |
| `sort_landmark.py` | alternative experiment: k-means the landmarks into 3 groups and crop 144x72x40 sub-volumes per group (`sort_three_region/G3`). This is where the `target_size = (144, 72, 40)` left in the old training script came from; it was **not** used for the published model |
| `whole_heatmap_save.py` | writes the max-merged heat-map of every case for inspection |
| `result_process.py` | prints connected-component centroids of predicted heat-maps (threshold 0.2) |
| `transform_to_nii.py` | converts one NRRD to NIfTI |

The original versions of the files that *were* re-organised (e.g. `UNet_xhy.py`, `tooth_region_pred.py`,
`2_train_and_valid.py`, `json_process_test0.py`) can be retrieved from the git history
(`git log --follow -- <new path>`).
