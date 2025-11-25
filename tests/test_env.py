from datetime import datetime
from zoneinfo import ZoneInfo
from configs import BACKEND_DIR


def test_env():
    print(BACKEND_DIR)


def test_now():
    print(datetime.now(ZoneInfo("Asia/Shanghai")))
    print(datetime.now())
    print(datetime.now().strftime("%Y%m%d_%H%M%S"))
