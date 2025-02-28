#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import time
import json
import argparse
import numpy as np
import sophon.sail as sail
from postprocess_numpy import PostProcess
from utils import COLORS, COCO_CLASSES
import logging
import cv2

logging.basicConfig(level=logging.INFO)

class Yolact():
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))

        # self.handle = self.net.get_handle()
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]

        # get input
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype= self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        # get output
        self.output_names = self.net.get_output_names(self.graph_name)
        if len(self.output_names) != 4:
            raise ValueError('only suport 4 outputs, but got {} outputs bmodel'.format(len(self.output_names)))

        self.output_tensors = {}
        self.output_scales = {}
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            output_scale = self.net.get_output_scale(self.graph_name, output_name)
            output = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
            self.output_tensors[output_name] = output
            self.output_scales[output_name] = output_scale

        # check batch size
        self.batch_size = self.input_shape[0]
        suppoort_batch_size = [1, 4]
        if self.batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(suppoort_batch_size, self.batch_size))
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # init preprocess
        self.use_vpp = False
        mean_bgr = np.array([103.94, 116.78, 123.68], dtype=np.float32)
        std_bgr = np.array([57.38, 57.12, 58.40], dtype=np.float32)
        self.mean = mean_bgr[::-1]  # bmcv use mean_rgb after bgr2rgb
        self.std = std_bgr[::-1]    # bmcv use std_rgb after bgr2rgb
        a = 1 / self.std
        b = - self.mean / self.std
        
        self.ab = tuple([(ia * self.input_scale, ib * self.input_scale) for ia, ib in zip(a, b)])

        # init postprocess
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.keep_top_k = args.keep_top_k

        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            keep_top_k=self.keep_top_k,
        )

        # init time
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess_bmcv(self, input_bmimg):
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                          sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        resized_img_rgb = self.resize_bmcv(rgb_planar_img)
        
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, sail.Format.FORMAT_RGB_PLANAR,
                                              self.img_dtype)
        
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, self.ab)
        return preprocessed_bmimg

    def resize_bmcv(self, bmimg):
        """
        resize for single sail.BMImage
        :param bmimg:
        :return: a resize image of sail.BMImage
        """
        img_w = bmimg.width()
        img_h = bmimg.height()
        r_w = self.net_w / img_w
        r_h = self.net_h / img_h
        resized_img_rgb = self.bmcv.resize(bmimg, self.net_w, self.net_h,  sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        return resized_img_rgb

    def predict(self, input_tensor, img_num):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            input_tensor:
        Returns:
        """
        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)
        outputs_dict = {}
        for name in self.output_names:
            # outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num] * self.output_scales[name]
            outputs_dict[name] = self.output_tensors[name].asnumpy()[:img_num]
        # resort
        out_keys = list(outputs_dict.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [outputs_dict[out_keys[i]] for i in ord]
        return out

    def __call__(self, bmimg_list):
        img_num = len(bmimg_list)
        ##################
        ori_size_list = []
        ##################
        if self.batch_size == 1:
            ori_h, ori_w = bmimg_list[0].height(), bmimg_list[0].width()
            ori_size_list.append((ori_w, ori_h))
            start_time = time.time()
            preprocessed_bmimg = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time

            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)

        else:
            BMImageArray = eval('sail.BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                ori_h, ori_w = bmimg_list[i].height(), bmimg_list[i].width()
                ori_size_list.append((ori_w, ori_h))
                start_time = time.time()
                preprocessed_bmimg = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                bmimgs[i] = preprocessed_bmimg.data()
            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)

        outputs = self.predict(input_tensor, img_num)

        self.inference_time += time.time() - start_time

        start_time = time.time()

        results = self.postprocess.infer_batch(outputs, ori_size_list)

        self.postprocess_time += time.time() - start_time

        return results

def draw_bmcv(handle, bmcv, bmimg, boxes, masks = None, classes_ids=None, conf_scores=None, save_path="", videos = False):
    
    img_bgr_planar = bmcv.convert_format(bmimg)
    # bm image -> image
    image = bmcv.bm_image_to_tensor(img_bgr_planar).asnumpy()[0]
    image = np.transpose(image, (1,2,0)).copy()

    draw_numpy(image, boxes, masks, classes_ids, conf_scores, videos)
    
    if save_path != None:
        save_name = save_path.split('/')[-1]
        cv2.imencode('.jpg', image)[1].tofile('{}.jpg'.format(save_path))
    #print('{}.jpg is saved.'.format(save_name))
    
  
def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None, videos=False):
    for idx in range(len(boxes)):
        left, top, width, height = boxes[idx, :].astype(np.int32).tolist()
        
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) % len(COLORS)]
        else:
            color = (0, 0, 255)
        
        thickness = 2  # Bounding box line thickness
        
        cv2.rectangle(image, (left, top), (left + width, top + height), color, thickness=thickness)

        if masks is not None:
            mask = masks[:, :, idx]
            class_id = int(classes_ids[idx]) % len(COLORS)
            color = COLORS[class_id]
            
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
            
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            
            text = COCO_CLASSES[classes_ids[idx] + 1] + ':' + str(round(conf_scores[idx], 2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8  # Text size
            text_thickness = 2  # Text line thickness
            
            # Calculate text position
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
            text_position = (left, top + height - 10)
            
            # Ensure text does not go beyond image boundary
            if text_position[0] + text_width > image.shape[1]:
                text_position = (left, top - 5)
            
            cv2.putText(image, text, text_position, font, font_scale, (0, 255, 0), thickness=text_thickness)

def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)

    # initialize net
    yolact = Yolact(args)
    batch_size = yolact.batch_size

    handle = sail.Handle(args.dev_id)
    bmcv = sail.Bmcv(handle)

    # warm up 
    # bmimg = sail.BMImage(handle, 550, 550, sail.Format.FORMAT_YUV420P, sail.DATA_TYPE_EXT_1N_BYTE)
    # for i in range(10):
    #     results = yolact([bmimg])
    
    yolact.init()

    decode_time = 0.0
    
    # test images
    if os.path.isdir(args.input):
        bmimg_list = []
        filename_list = []
        results_list = []
        cn = 0

        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1

                logging.info("{}, img_file: {}".format(cn, img_file))
                # decode
                start_time = time.time()
                decoder = sail.Decoder(img_file, True, args.dev_id)
                bmimg = sail.BMImage()
                ret = decoder.read(handle, bmimg)
                # print(bmimg.format(), bmimg.dtype())
                if ret != 0:
                    logging.error("{} decode failure.".format(img_file))
                    continue
                decode_time += time.time() - start_time

                bmimg_list.append(bmimg)
                filename_list.append(filename)
                if (len(bmimg_list) == batch_size or cn == len(filenames)) and len(bmimg_list):
                    # predict

                    results = yolact(bmimg_list)
                    for i, filename in enumerate(filename_list):
                        # det[0]: class_id, det[1]:conf_score, det[2]:boxes, det[3]:masks
                        det = [results[j][i] for j in range(4)]
                        if len(det[-1]) == 0:
                            print("None Object in {}".format(filename))
                            continue
                            
                        # save image
                        save_path = os.path.join(output_img_dir, filename)
                        #         bmcv,bmimage,box,class_id,conf_score,save_path
                        draw_bmcv(handle, bmcv, bmimg_list[i], det[2], det[3], classes_ids=det[0], conf_scores=det[1],
                                  save_path=save_path)
                        
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['bboxes'] = []
                        for idx in range(det[0].shape[0]):
                            bbox_dict = dict()
                            # x, y ,w, h
                            x1, y1, x2, y2 = det[2][idx]
                            score = det[1][idx]
                            category_id = det[0][idx]

                            x1 = max(x1, 0)
                            y1 = max(y1, 0)
                            if x2 < 0 or y2 < 0:
                                continue
                                
                            bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2, 3)),
                                                 float(round(y2, 3))]
                            bbox_dict['category_id'] = int(category_id)
                            bbox_dict['score'] = float(round(score, 5))
                            res_dict['bboxes'].append(bbox_dict)
                        results_list.append(res_dict)

                    bmimg_list.clear()
                    filename_list.clear()
        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[
            -1] + "_bmcv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    # # test video
    else:
        decoder = sail.Decoder(args.input, True, args.dev_id)
        if not decoder.is_opened():
            raise Exception("can not open the video")
        video_name = os.path.splitext(os.path.split(args.input)[1])[0]
        cn = 0
        frame_list = []
        end_flag = False
        while not end_flag:
            frame = sail.BMImage()
            start_time = time.time()
            ret = decoder.read(handle, frame)
            decode_time += time.time() - start_time

            if ret:
                end_flag = True
            else:
                frame_list.append(frame)
            if (len(frame_list) == batch_size or end_flag) and len(frame_list):
                results = yolact(frame_list)
                for i, frame in enumerate(frame_list):
                    det = [results[j][i] for j in range(4)]

                    if len(det[-1]) == 0:
                        print("None object in this frame.")
                        continue
                    cn += 1

                    logging.info("{}, det nums: {}".format(cn, det[0].shape[0]))
                    save_path = os.path.join(output_img_dir, video_name + '_' + str(cn) + '.jpg')
                    
                    draw_bmcv(handle, bmcv, frame_list[i], det[2], det[3], classes_ids=det[0], conf_scores=det[1],
                          save_path=save_path)

                frame_list.clear()
        decoder.release()
        logging.info("result saved in {}".format(output_img_dir))

    # calculate speed  
    if cn > 0:
        logging.info("------------------ Predict Time Info ----------------------")
        decode_time = decode_time / cn
        preprocess_time = yolact.preprocess_time / cn
        inference_time = yolact.inference_time / cn
        postprocess_time = yolact.postprocess_time / cn
        logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
        # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
        # qps = 1 / average_latency
        # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))              


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/yolact_bm1684x_fp32_1b.bmodel',
                        help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--keep_top_k', type=int, default=100, help='keep top k candidate boxs')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
