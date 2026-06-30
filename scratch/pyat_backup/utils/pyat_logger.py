#! /usr/bin/env python3
# coding: utf-8

import logging

# pylint:disable=consider-using-f-string
FORMAT = "[ {} ] {}: {}".format("%(levelname)s", "%(asctime)s", "%(message)s")
logging.basicConfig(level=logging.INFO, format=FORMAT)

LEN_LOG = 60
NO = ":    NO"
OK = ": OK"


def info_progress_layer(logger, msg, layer, c, n):
    logger.info("{} {}/{}".format("{0:.<{size}}".format("Process dtm " + msg + " " + layer, size=LEN_LOG), c, n))


def info_progress(logger, msg, c, n):
    logger.info("{} {}/{}".format("{0:.<{size}}".format(msg, size=LEN_LOG), c, n))


logger = logging.getLogger("Global Logger")


def info( msg, *args, **kwargs):
    """proxy to logger.info using global logger"""
    logger.info( msg, *args, **kwargs)

def warning( msg, *args, **kwargs):
    """proxy to logger.info using global logger"""
    logger.warning( msg, *args, **kwargs)

def error( msg, *args, **kwargs):
    """proxy to logger.info using global logger"""
    logger.error( msg, *args, **kwargs)

def debug( msg, *args, **kwargs):
    """proxy to logger.info using global logger"""
    logger.debug( msg, *args, **kwargs)
