import cv2
import time
from collections import Counter
import argparse
from utils import EasyocrNumberPlateRecognition
from pymot.object_detection import yolo
from pymot.mot import MOT


def license_plate_recognition(
    frame, tracking_id, obj_coord, **proc_kwargs
) -> tuple[dict, tuple]:
    """License plate recognition function for a MOT tracker"""

    xmin, ymin, xmax, ymax = obj_coord

    # If number of processed objects is more than max limit,
    # delete the first to enter.
    if len(proc_kwargs["proc_obj_info"]) > proc_kwargs["max_n_obj"]:
        if proc_kwargs["last_obj_to_del"] not in proc_kwargs["proc_obj_info"]:
            proc_kwargs["last_obj_to_del"] += 1
        else:
            del proc_kwargs["proc_obj_info"][proc_kwargs["last_obj_to_del"]]

    # If there are no processed objects set the first to enter id
    # as last_obj_to_del.
    if len(proc_kwargs["proc_obj_info"]) == 0:
        proc_kwargs["last_obj_to_del"] = tracking_id

    # If new tracking id, create new entry in the processed objects dict.
    if tracking_id > proc_kwargs["prev_id"]:
        proc_kwargs["prev_id"] = tracking_id
        proc_kwargs["proc_obj_info"][tracking_id] = [1, [], [], False]

    # If the number of performed processings on a object with a certain
    # tracking id is less than needed, do another processing.
    if proc_kwargs["proc_obj_info"][tracking_id][0] <= proc_kwargs["n_det"]:
        bbox_frame = frame[ymin:ymax, xmin:xmax]
        plate_number = ""

        (cl_ids, probs, bboxes) = proc_kwargs["model_lp"].detect(bbox_frame)
        for i, box in enumerate(bboxes):
            if cl_ids[i] == 0:
                bbox_x, bbox_y, bbox_w, bbox_h = box[0], box[1], box[2], box[3]

                bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = (
                    bbox_x,
                    bbox_y,
                    bbox_x + bbox_w,
                    bbox_y + bbox_h,
                )

                num_plate_img = bbox_frame[
                    bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax
                ]

                plate_number = proc_kwargs["ocr"].recognize_plate_number(
                    num_plate_img, (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
                )
                break

        proc_kwargs["proc_obj_info"][tracking_id][1].append(plate_number)

        idx = int(
            proc_kwargs["proc_obj_info"][tracking_id][0]
            * (len(proc_kwargs["proc_animation"]) / proc_kwargs["n_det"])
        )

        proc_kwargs["proc_obj_info"][tracking_id][2] = proc_kwargs[
            "proc_animation"
        ][idx - 1 if idx == len(proc_kwargs["proc_animation"]) else idx]

        proc_kwargs["proc_obj_info"][tracking_id][0] += 1

    # If the number of performed processings on a object with a certain
    # tracking id has sufficed do final processing.
    if (
        proc_kwargs["proc_obj_info"][tracking_id][0]
        == proc_kwargs["n_det"] + 1
    ):
        temp_buf = list(
            filter(None, proc_kwargs["proc_obj_info"][tracking_id][1])
        )
        temp_buf = Counter(temp_buf)

        if len(temp_buf) > 0:
            plate_number = temp_buf.most_common(1)[0][0]
            plate_number = plate_number.upper()
            proc_kwargs["proc_obj_info"][tracking_id][2] = plate_number

            proc_kwargs["proc_obj_info"][tracking_id][3] = True

            if proc_kwargs["save_lp"]:
                cv2.imwrite(
                    proc_kwargs["num_plate_data_dir"] + plate_number + ".jpg",
                    num_plate_img,
                )
        else:
            proc_kwargs["proc_obj_info"][tracking_id][2] = "NO LP"

        proc_kwargs["proc_obj_info"][tracking_id][0] += 1

    return (
        (
            proc_kwargs["proc_obj_info"],
            proc_kwargs["last_obj_to_del"],
            proc_kwargs["prev_id"],
        ),
        (
            False,
            proc_kwargs["proc_obj_info"][tracking_id][3],
            ((0, 255, 0), (0, 0, 255)),
            proc_kwargs["proc_obj_info"][tracking_id][2],
        ),
    )


def yolo_object_tracking_with_apps(args: argparse.Namespace) -> None:
    with open("yolo_data/mscoco_classes.txt") as f:
        classes = f.read().splitlines()

    track_classes = ["car", "bus", "truck", "motorcycle"]

    mot_cfg = {
        "od_classes": classes,
        "od_algo": "yolo",
        "od_wpath": "yolo_data/yolov4-tiny.weights",
        "od_cpath": "yolo_data/yolov4-tiny.cfg",
        "od_nms_thr": 0.4,
        "od_conf_thr": 0.5,
        "od_img_size": 416,
        "od_cuda": True,
        "t_classes": track_classes,
        "t_algo": "deepsort",
        "t_cuda": True,
        "t_metric": "cosine",
        "t_max_cosine_distance": 0.2,
        "t_budget": 100,
        "t_max_iou_distance": 0.7,
        "t_max_age": 70,
        "t_n_init": 3,
    }

    obj_tracking = MOT(mot_cfg)

    if args.video_file_path is not None:
        cap = cv2.VideoCapture(args.video_file_path)
    else:
        cap = cv2.VideoCapture(0)

    prev_id = -1
    n_det = 10
    max_n_obj = 50
    last_obj_to_del = 0
    proc_obj_info = {}
    proc_animation = {
        0: "|",
        1: "|" * 2,
        2: "|" * 3,
        3: "|" * 4,
        4: "|" * 5,
        5: "|" * 6,
        6: "|" * 7,
        7: "|" * 8,
    }

    # License plate recognition
    ocr = EasyocrNumberPlateRecognition(area_th=0.2)

    num_plate_data_dir = "saved_number_plates/"

    model_lp = yolo.YOLO(
        "yolo_data/yolov4-tiny_lp.weights",
        "yolo_data/yolov4-tiny_1_cl.cfg",
        nms_thr=0.4,
        conf_thr=0.5,
        img_size=416,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break

        t1 = time.time()

        (
            t_ojb_info,
            frame_with_bboxes,
            proc_obj_info_tuple,
        ) = obj_tracking.track_objects(
            frame,
            license_plate_recognition,
            prev_id=prev_id,
            n_det=n_det,
            max_n_obj=max_n_obj,
            last_obj_to_del=last_obj_to_del,
            proc_animation=proc_animation,
            ocr=ocr,
            proc_obj_info=proc_obj_info,
            num_plate_data_dir=num_plate_data_dir,
            model_lp=model_lp,
            save_lp=args.lpr_save_img,
        )

        if None not in proc_obj_info_tuple:
            proc_obj_info, last_obj_to_del, prev_id = proc_obj_info_tuple

        fps = int(1 / (0.0001 + time.time() - t1))

        cv2.putText(
            frame_with_bboxes,
            "FPS: {}".format(fps),
            (0, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.imshow("License plate recognition", frame_with_bboxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License plate recognition")
    parser.add_argument(
        "--lpr_save_img",
        type=bool,
        default=False,
        help="Whether or not to save license plate image",
    )
    parser.add_argument(
        "-v",
        "--video_file_path",
        default=None,
        help="Use a video for tracking and if the path is not provided use a webcam",
    )
    args = parser.parse_args()

    yolo_object_tracking_with_apps(args)
