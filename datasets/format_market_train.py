"""
   Format Market-1501 training images with consecutive labels.

   This code modifies the data preparation method of
   "Learning Deep Feature Representations with Domain Guided Dropout for Person Re-identification".

"""

import shutil
from glob import glob
from datasets.utils import *


def _format_train_data(in_dir, output_dir):
    # cam_0 to cam_5
    for i in xrange(6):
        mkdir_if_missing(osp.join(output_dir, 'cam_' + str(i)))
    # pdb.set_trace()
    images = glob(osp.join(in_dir, '*.jpg'))
    images.sort()
    identities = []
    prev_pid = -1
    for name in images:
        name = osp.basename(name)
        p_id = int(name[0:4])
        c_id = int(name[6]) - 1
        if prev_pid != p_id:
            identities.append([])
            prev_cid = -1
        p_images = identities[-1]
        if prev_cid != c_id:
            p_images.append([])
        v_images = p_images[-1]
        file_name = 'cam_{}/cam_{:02d}_{:05d}_{:05d}.jpg'.format(c_id, c_id, len(identities)-1, len(v_images))
        shutil.copy(osp.join(in_dir, name),
                    osp.join(output_dir, file_name))
        v_images.append(file_name)
        prev_pid = p_id
        prev_cid = c_id
    # Save meta information into a json file
    meta = {'name': 'market1501', 'shot': 'multiple', 'num_cameras': 6}
    meta['identities'] = identities
    write_json(meta, osp.join(output_dir, 'meta.json'))
    num_images = len(images)
    num_classes = len(identities)
    print("Training data has %d images of %d classes" % (num_images, num_classes))


def run(image_dir):
    """Format the datasets with consecutive labels.

    Args:
        image_dir: The dataset directory where the raw images are stored.

    """
    in_dir = image_dir + "_raw"
    os.rename(image_dir, in_dir)
    mkdir_if_missing(image_dir)
    _format_train_data(in_dir, image_dir)
