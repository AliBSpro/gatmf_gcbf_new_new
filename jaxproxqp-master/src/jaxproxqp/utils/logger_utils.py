import sys

# 兼容性导入 - 如果loguru不可用則使用标准日誌
try:
    from loguru import logger
    HAS_LOGURU = True
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    HAS_LOGURU = False

FORMAT = "<level>{level.name}</level> | " "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


def set_logger_style():
    if HAS_LOGURU:
        logger.remove()
        logger.add(sys.stdout, format=FORMAT, level="TRACE", filter=lambda record: "compile_log" not in record["extra"])
    else:
        # 标准日誌不需要特殊设置
        pass
