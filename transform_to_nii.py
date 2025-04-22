import SimpleITK as sitk
import os

nrrd_image_filename = \
    [x for x in os.listdir(os.path.join('/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData/092_sunyimeng'))
     if x.endswith('.nrrd')][0]
nrrd_image_file_path = os.path.join('/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData/092_sunyimeng',
                                    nrrd_image_filename)
sitk_image = sitk.ReadImage(nrrd_image_file_path)
sitk.WriteImage(sitk_image, '/home/user16/sharedata/GXE/SkullWidth/data/imageStandardData/092_sunyimeng/091.nii.gz')
