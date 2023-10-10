import logging
import os

# Directly modified from the OT-flow Github Depository

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False, mode="a"):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode=mode)
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def makedirs(dirname):
    """
    make the directory folder structure
    :param dirname: string path
    :return: void
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
