# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime


def get_logger(log_dir: str, name: str = "MambaTSAD"):
    """
    简单封装 logging，日志同时打印到终端和文件。

    参数
    ----
    log_dir:
        日志文件保存目录。
    name:
        logger 名称（同一进程内保持唯一即可）。
    """
    os.makedirs(log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{time_str}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 清空旧 handler，防止在 Jupyter / 多次调用时重复打印
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件输出
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"日志已写入：{log_file}")
    return logger
