# weights/

Git-ignored folder for checkpoints **you train yourself** with `scripts/4_train_landmark.py`
(stage 2) and your own tooth-region model (stage 1).

The checkpoints used in the paper, `model/tooth_best_model/bestmodel.pth` (stage 1,
`UNet3D_simple(n_class=1)`, plain `state_dict`) and `model/UNet3d_stage1/Unet_model08.pt`
(stage 2, `UNet3d(n_class=1, act='relu')`, `{'epoch','state_dict','optimizer_state_dict'}`),
**are not available**: they were never committed and the only copies were lost with the lab
server storage. See the main README, section "Pretrained weights".

`cbct_landmark.models.load_checkpoint` accepts both formats.
