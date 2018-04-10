from ctypes import *
import argparse
import cv2
import os
import glob
import random
import json


def get_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', nargs='?')
    parser.add_argument('--image', nargs='?')
    return parser


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

draw_detection = lib.draw_detections
"""
predict_image.argtypes = [IMAGE, POINTER(DETECTION)]
predict_image.restype = POINTER(c_float)

save_image = lib.save_image
save_image.a

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE
"""



def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def convert_coords(box):
    center_x = box.x
    center_y = box.y
    box.x = center_x - (box.w / 2)
    box.y = center_y - (box.h / 2)
    return box


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    image = image.decode("utf-8")
    num = c_int(0)
    pnum = pointer(num)
    ou = predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    imgcv = cv2.imread(image)
    h, w, _ = imgcv.shape
    #res = []
    colors = {}
    result = []
    resultsForJSON = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                name = meta.names[i].decode("utf-8")
                if colors.get(name) is None:
                    colors[name] = get_color()
                b = dets[j].bbox
                b = convert_coords(b)
                thick = int((h + w) // 350)
                resultsForJSON.append(
                    {"label": name, "confidence": dets[j].prob[i], "topleft": {"x": b.x, "y": b.y},
                     "bottomright": {"x": b.x + b.w, "y": b.y + b.h}})

                cv2.rectangle(imgcv, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), colors.get(name), thick)
                font_size = 1e-3 * h / 2
                print(font_size)
                cv2.putText(imgcv, name, (int(b.x), int(b.y - font_size * 10)), 0, font_size, colors.get(name), thick // 3)

    outfolder = os.path.join(os.path.dirname(image), 'out')
    if os.path.exists(outfolder) is False:
        os.mkdir(outfolder)
    img_name = os.path.join(outfolder, os.path.basename(image))
    textJSON = json.dumps(resultsForJSON)
    textFile = os.path.splitext(img_name)[0] + ".json"
    with open(textFile, 'w') as f:
        f.write(textJSON)
    cv2.imwrite(img_name, imgcv)
    free_image(im)
    free_detections(dets, num)
    return result


def detect_folder(net, meta, folder):
    files = [os.path.basename(x) for x in glob.glob(folder + "/*[.jpg, .jpeg, .JPG, .JPEG]")]
    result = []
    for image in files:
        res_img = {}
        res_img['image'] = image
        res_img['result'] = detect(net, meta, bytes(os.path.join(folder, image), 'utf-8'))
        result.append(res_img)
    return result


if __name__ == "__main__":
    parser = create_parser()
    root = parser.parse_args()
    net = load_net(b"bdf.cfg", b"bdf_final.weights", 0)
    meta = load_meta(b"bdf.data")
    if root.folder:
        folder = root.folder
        r = detect_folder(net, meta, folder)
    elif root.image:
        r = detect(net, meta, bytes(root.image, "utf-8"))
    else:
        exit(0)
    result = json.dumps(r)
    print(result)

