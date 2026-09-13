# Pretrained weights

Place the two checkpoints used in the paper here (they are not tracked by git, see `.gitignore`):

| file | original name / path | network | format |
|---|---|---|---|
| `tooth_region_bestmodel.pth` | `model/tooth_best_model/bestmodel.pth` | `UNet3D_simple(n_class=1)` (stage 1) | plain `state_dict` |
| `landmark_Unet_model08.pt` | `model/UNet3d_stage1/Unet_model08.pt` | `UNet3d(n_class=1, act='relu')` (stage 2) | `{'epoch', 'state_dict', 'optimizer_state_dict'}` |

`cbct_landmark.models.load_checkpoint` accepts both formats. Download links are listed in the
main README under "Pretrained weights".
