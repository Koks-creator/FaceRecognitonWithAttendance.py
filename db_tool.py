from dataclasses import dataclass
from typing import List
import sqlite3


@dataclass()
class DBtool:
    db_path: str

    def __post_init__(self):
        self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        self.c = self.conn.cursor()

        sql = "CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, name VARCHAR(50), age INT(10)," \
              " position VARCHAR(80), sex VARCHAR(10), image BLOB)"
        self.c.execute(sql)

    def fetch_data(self) -> List[dict]:
        table_records = []
        with self.conn:
            self.c.execute("SELECT * FROM faces")
            data = self.c.fetchall()
            for row in data:
                record = {"Id": row[0], "Name": row[1], "Age": row[2], "Position": row[3], "Sex": row[4], "ImageBytes": row[5]}
                table_records.append(record)
            return table_records

    def add_data(self, name: str, age: int, position: str, sex: str, img_path: str) -> bool:
        with self.conn:
            im = open(img_path, 'rb').read()

            self.c.execute("INSERT INTO faces (name, age, position, sex, image) VALUES (?, ?, ?, ?, ?)",
                           (name, age, position, sex, sqlite3.Binary(im)))
            if self.c.rowcount != 0:
                self.conn.commit()
                return True
            return False

    def delete_row(self, record_id: int) -> bool:
        with self.conn:
            self.c.execute(f"DELETE FROM faces WHERE id={record_id}")
            if self.c.rowcount != 0:
                self.conn.commit()
                return True
            return False

    def delete_all(self) -> bool:
        with self.conn:
            self.c.execute("DELETE FROM faces")
            self.conn.commit()
        return True


if __name__ == '__main__':
    face_reg = DBtool(db_path="faces.db")
    # print(face_reg.delete_all())
    print(face_reg.add_data("Thom Yorke", 53, "Musician", "Male", r"images2/thom.jpg"))
    data = face_reg.fetch_data()
    print(data)
    # image_raw = data[0][5]
    # decoded = cv2.imdecode(np.frombuffer(image_raw, np.uint8), -1)
    # print(decoded)
    # cv2.imshow("res", decoded)
    # cv2.waitKey(0)



