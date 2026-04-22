import time
import random

class MockSparkStreaming:
    """Giả lập việc đọc dữ liệu từ YouTube Live Chat qua Spark"""
    def __init__(self):
        self.mock_comments = [
            "Video hay quá anh ơi, hóng phần tiếp theo! ❤️",
            "Cái thằng này nói chuyện như clgt, đấm cho phát giờ 😡",
            "Mọi người ơi mình mới mua đt xịn xò lắm luôn 📱",
            "Giỏi quá vcl cả họ tự hào luôn nhé smirk",
            "Bắc kỳ lại bắt đầu gáy rồi đấy...",
            "Ét ô ét cứu tôi với mng ơi 🆘",
            "Phim này xem phí thời gian thực sự, rác rưởi."
        ]

    def stream_generator(self):
        while True:
            yield random.choice(self.mock_comments)
            time.sleep(3) # Cứ 3 giây có 1 comment mới
if __name__ == '__main__':
    print('hi')

# py -m app.inference_service.streaming_engine