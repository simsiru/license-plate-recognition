import cv2
import numpy as np
import psycopg2 as pg
import re
import easyocr
import io
import pandas as pd


def draw_bbox_tracking(
    coords,
    height,
    frame,
    tracking_id,
    class_name="",
    rand_colors=True,
    rec_bool=False,
    colors=((0, 255, 0), (0, 0, 255)),
    info_text="",
    unknown_obj_info="UNKNOWN",
):
    """Function for visualizing tracking bounding boxes"""

    assert len(colors) == 2, "Colors must a list of tuples of length 2"

    xmin, ymin, xmax, ymax = coords

    if rand_colors:
        np.random.seed(tracking_id)
        r = np.random.rand()
        g = np.random.rand()
        b = np.random.rand()
        color = (int(r * 255), int(g * 255), int(b * 255))
    else:
        if rec_bool:
            color = colors[0]
        else:
            info_text = unknown_obj_info
            color = colors[1]

    cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), color, 2)

    text = "{} ID: {} [{}]".format(class_name, tracking_id, info_text)

    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, 0.75, thickness=1
    )

    if (ymax + text_height) > height:
        cv2.rectangle(
            frame,
            (xmin, ymax),
            (xmin + text_width, ymax - text_height - baseline),
            color,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            frame,
            text,
            (xmin, ymax - 4),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    else:
        cv2.rectangle(
            frame,
            (xmin, ymax + text_height + baseline),
            (xmin + text_width, ymax),
            color,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            frame,
            text,
            (xmin, ymax + text_height + 3),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )


class EasyocrNumberPlateRecognition:
    """Class for interface with easyOCR for license plate recognition"""

    def __init__(self, area_th=0.2):
        self.easyocr_reader = easyocr.Reader(["en"], gpu=True)
        self.letter_plate_area_ratio = area_th

    def recognize_plate_number(self, num_plate_box, coords):
        num_plate_box = cv2.cvtColor(num_plate_box, cv2.COLOR_BGR2RGB)

        xmin, ymin, xmax, ymax = coords

        plate_num = ""

        box_area = (xmax - xmin) * (ymax - ymin)

        try:
            text = self.easyocr_reader.readtext(num_plate_box, detail=1)

            for res in text:
                length = np.sum(np.subtract(res[0][1], res[0][0]))
                height = np.sum(np.subtract(res[0][2], res[0][1]))

                if (
                    (length * height) / box_area
                ) > self.letter_plate_area_ratio:
                    plate_num += res[1]

            plate_num = re.sub("[\W_]+", "", plate_num)
        except:
            text = None

        return plate_num


class DBInterface:
    """Class for interface with PostgreSQL database"""

    def __init__(
        self,
        password="148635",
        username="postgres",
        hostname="localhost",
        database="face_recognition",
        port_id=5432,
    ):
        self.host = hostname
        self.dbname = database
        self.username = username
        self.password = password
        self.port = port_id

    def execute_sql_script(
        self, sql_script, values_insert=None, return_result=False
    ):
        conn = None
        cur = None
        df = None

        try:
            with pg.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.username,
                password=self.password,
                port=self.port,
            ) as conn:

                if return_result:
                    df = pd.read_sql_query(sql_script, conn)
                else:
                    with conn.cursor() as cur:
                        if values_insert is not None:
                            cur.execute(sql_script, values_insert)
                        else:
                            cur.execute(sql_script)

        except Exception as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()

        if return_result:
            return df

    def numpy_array_to_bytes(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return out.read()

    def bytes_to_numpy_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
