from loguru import logger


def test_log():
    try:
        1 / 0
    except Exception as e:
        print("\n")
        logger.error("计算出错, 详细信息: {}", e)
        print("\n")
        logger.error("计算出错, 详细信息: ", e)
        print("\n")
        logger.opt(exception=True).error("计算出错, 详细信息")
        print("\n")
        logger.error("计算出错, 详细信息", exc_info=e)
        print("\n")
        logger.error("计算出错, 详细信息", exc_info=True)
        print("\n")
        logger.exception("计算出错, 详细信息")
