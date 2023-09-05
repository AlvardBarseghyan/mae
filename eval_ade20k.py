import os
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np

# adapt from https://github.com/CSAILVision/placeschallenge/blob/master/sceneparsing/evaluationCode/utils_eval.py
def intersectionAndUnion(imPred, imLab, numClass):
	imPred = np.asarray(imPred)
	imLab = np.asarray(imLab)

	# Remove classes from unlabeled pixels in gt image. 
	# We should not penalize detections in unlabeled portions of the image.
	imPred = imPred * (imLab>0)

	# Compute area intersection:
	intersection = imPred * (imPred==imLab)
	(area_intersection,_) = np.histogram(intersection, bins=numClass, range=(1, numClass))

	# Compute area union:
	(area_pred,_) = np.histogram(imPred, bins=numClass, range=(1, numClass))
	(area_lab,_) = np.histogram(imLab, bins=numClass, range=(1, numClass))
	area_union = area_pred + area_lab - area_intersection
	
	return (area_intersection, area_union)

def pixelAccuracy(imPred, imLab):
	imPred = np.asarray(imPred)
	imLab = np.asarray(imLab)

	# Remove classes from unlabeled pixels in gt image. 
	# We should not penalize detections in unlabeled portions of the image.
	pixel_labeled = np.sum(imLab>0)
	pixel_correct = np.sum((imPred==imLab)*(imLab>0))
	pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

	return (pixel_accuracy, pixel_correct, pixel_labeled)


def get_pred_label_pairs(pred_folder, mask_folder):
    pred_paths = []
    mask_paths = []
    for filename in os.listdir(pred_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".png"):
            predpath = os.path.join(pred_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                pred_paths.append(predpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return pred_paths, mask_paths

def evaluate_pairs(pred_path):
    nclass = 150
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    # tbar = tqdm(zip(pred_paths, mask_paths))
    annots = np.load(pred_path, allow_pickle=True)
    tbar = tqdm(annots, total=len(annots))

    for annot in tbar:
        # load image
        # pred = np.array(Image.open(predpath))
        # mask = np.array(Image.open(maskpath))
        pred = annot['patch_labels']
        mask = annot['mask']

        inter, union = intersectionAndUnion(pred, mask, nclass)
        _, correct, labeled = pixelAccuracy(pred, mask)
        total_inter += inter
        total_union += union
        total_correct += correct
        total_label += labeled
        # display
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()
        tbar.set_description(
            'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))

if __name__ == "__main__":
    # check args
    # if len(sys.argv) < 3:
    #     sys.exit('Usage: % sprediction_folder groundtruth_folder' % sys.argv[0])
    # if not os.path.exists(sys.argv[1]):
    #     sys.exit('ERROR: prediction_folder %s was not found!' % sys.argv[1])
    # if not os.path.exists(sys.argv[2]):
    #     sys.exit('ERROR: groundtruth_folder %s was not found!' % sys.argv[2])


    # get image pair paths
    # pred_paths, mask_paths = get_pred_label_pairs(sys.argv[1], sys.argv[2])
    evaluate_pairs(sys.argv[1])