from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import mysql.connector
from datetime import datetime, timedelta
from fastapi_utils.tasks import repeat_every
import logging
import tensorflow as tf
import numpy as np

app = FastAPI()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

# Thông tin kết nối MySQL
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password1234',
    'database': 'stock_prediction'
}

# Đường dẫn đến thư mục chứa mô hình đã lưu
model_path = 'E:\\#UIT\\31-Pattern\\Project\\App-stock-prediction\\model\\Model_TKAN_S_HPG'

# Tải mô hình dự đoán
loaded_model = tf.saved_model.load(model_path)


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


# Kiểm tra xem bản ghi đã tồn tại hay chưa
def record_exists(cursor, ticker, timestamp):
    query = """
    SELECT 1 FROM hpg_table
    WHERE Ticker = %s AND Timestamp = %s
    LIMIT 1
    """
    cursor.execute(query, (ticker, timestamp))
    return cursor.fetchone() is not None


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


# Dự đoán phần trăm thay đổi
def predict_percentage(inputs):
    inputs = np.array(inputs).reshape((1, 4, 1)).astype(np.float32)
    y_pred_tensor = loaded_model(inputs)
    y_pred = y_pred_tensor.numpy().flatten()
    return y_pred[0]


# Lấy dữ liệu percentage từ database
def get_percentage_data(cursor, start_time):
    percentages = []
    for i in range(5):
        if i == 0:
            continue
        query = """
        SELECT Percentage FROM hpg_table
        WHERE Ticker = 'HPG' AND Timestamp = %s
        """
        timestamp = start_time - timedelta(minutes=i * 5)
        cursor.execute(query, (timestamp,))
        result = cursor.fetchone()
        logging.error(f"insert percentage data: {result}")
        if result:
            percentages.insert(0, result[0])  # Thêm vào đầu danh sách
        else:
            raise HTTPException(status_code=404, detail=f"No data found for time: {timestamp}")
    return percentages


# Tạo template
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict_stock_price(request: Request, prediction_time: str = Form(...)):
    try:
        prediction_time = datetime.strptime(prediction_time, '%H:%M').time()
        if prediction_time <= datetime.strptime('09:35', '%H:%M').time():
            return templates.TemplateResponse("form.html", {
                "request": request,
                "error_message": "Prediction time must be after 09:35"
            })

        current_time = datetime.now().time()
        if prediction_time <= (datetime.combine(datetime.today(), current_time) - timedelta(minutes=5)).time():
            return templates.TemplateResponse("form.html", {
                "request": request,
                "error_message": "Prediction time must be at least 5 minutes after the current time"
            })

        # Lấy dữ liệu từ DB
        current_date = datetime.now().date()
        prediction_datetime = datetime.combine(current_date, prediction_time)

        connection = get_db_connection()
        if connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        cursor = connection.cursor()

        percentages = get_percentage_data(cursor, prediction_datetime)

        # Dự đoán percentage
        predicted_percentage = predict_percentage(percentages)

        # Lấy giá close hiện tại
        query = """
        SELECT Close FROM hpg_table
        WHERE Ticker = 'HPG' AND Timestamp = %s
        """
        cursor.execute(query, (prediction_datetime - timedelta(minutes=5),))
        current_close = cursor.fetchone()
        if current_close:
            current_close = current_close[0]
        else:
            return templates.TemplateResponse("form.html", {
                "request": request,
                "error_message": "No close price found for the last 5 minutes"
            })

        # Tính giá close dự đoán
        predicted_close = current_close * (1 + predicted_percentage / 100)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "prediction_time": prediction_time,
            "predicted_percentage": predicted_percentage,
            "predicted_close": predicted_close
        })
    except ValueError:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error_message": "Invalid time format. Use HH:MM"
        })
    except HTTPException as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error_message": e.detail
        })
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error_message": "An unexpected error occurred. Please try again later."
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
