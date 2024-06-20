from fastapi import FastAPI
from pydantic import BaseModel
import requests
import mysql.connector
from datetime import datetime, timedelta
from fastapi_utils.tasks import repeat_every
import logging

app = FastAPI()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

# Thông tin kết nối MySQL
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'stock_prediction'  # Đảm bảo tên cơ sở dữ liệu chính xác
}


# Kết nối đến MySQL
def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except mysql.connector.Error as err:
        logging.error(f"Error connecting to MySQL: {err}")
        return None


# Mô hình dữ liệu
class StockData(BaseModel):
    Ticker: str
    Timestamp: datetime
    Close: float
    Prev_Close: float = None
    Percentage: float = None


# Lấy giá close của ngày hôm trước
def get_prev_close(cursor, ticker, current_date):
    previous_day = current_date - timedelta(days=1)
    query = """
    SELECT Close FROM hpg_table
    WHERE Ticker = %s AND DATE(Timestamp) = %s
    ORDER BY Timestamp DESC
    LIMIT 1
    """
    cursor.execute(query, (ticker, previous_day))
    result = cursor.fetchone()
    if result:
        return result[0]
    return None


# Kiểm tra xem bản ghi đã tồn tại hay chưa
def record_exists(cursor, ticker, timestamp):
    query = """
    SELECT 1 FROM hpg_table
    WHERE Ticker = %s AND Timestamp = %s
    LIMIT 1
    """
    cursor.execute(query, (ticker, timestamp))
    return cursor.fetchone() is not None


# Xử lý và lưu dữ liệu vào MySQL
def process_and_store_data():
    url = 'https://banggia.cafef.vn/stockhandler.ashx?center=undefined'
    response = requests.get(url)

    if response.status_code != 200:
        logging.error(f"Failed to fetch data, status code: {response.status_code}")
        return

    data = response.json()

    connection = get_db_connection()
    if connection is None:
        return
    cursor = connection.cursor()

    for item in data:
        if item['a'] == 'HPG':
            ticker = item['a']
            close_price = float(item['l'])
            timestamp = datetime.strptime(item['Time'], '%H:%M %d/%m/%Y')
            current_date = timestamp.date()

            if record_exists(cursor, ticker, timestamp):
                logging.info(f"Record already exists for {ticker} at {timestamp}")
                continue

            prev_close = get_prev_close(cursor, ticker, current_date)
            if prev_close is not None:
                percentage = ((close_price - prev_close) / prev_close) * 100
            else:
                percentage = None

            stock_data = StockData(
                Ticker=ticker,
                Timestamp=timestamp,
                Close=close_price,
                Prev_Close=prev_close,
                Percentage=percentage
            )

            insert_query = """
            INSERT INTO hpg_table (Ticker, Timestamp, Close, Prev_Close, Percentage)
            VALUES (%s, %s, %s, %s, %s)
            """
            try:
                cursor.execute(insert_query, (
                stock_data.Ticker, stock_data.Timestamp, stock_data.Close, stock_data.Prev_Close,
                stock_data.Percentage))
                logging.info(f"Inserted data: {stock_data}")
            except mysql.connector.Error as err:
                logging.error(f"Error inserting data: {err}")

    connection.commit()
    cursor.close()
    connection.close()


@app.on_event("startup")
@repeat_every(seconds=60)  # Gọi hàm này mỗi phút
def fetch_and_store_data():
    process_and_store_data()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
