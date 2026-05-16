import json
import requests
import pyat.utils.pyat_logger as log


def retrieve_depth_emodnet(longitude, latitude):
    """Allow to retrieve a depth value from EMODnet web services"""
    logger = log.logging.getLogger(__file__)
    r = requests.get(f"https://rest.emodnet-bathymetry.eu/depth_sample?geom=POINT({longitude} {latitude}) ", timeout=10)

    if r.status_code != 200:
        logger.error(f"Error while querying data {r.status_code} : {r.reason}")
        return float("nan")
    data = json.loads(r.text)
    if "avg" in data:
        return float(data["avg"])
    return float(data)


def retrieve_depth_gmrt(longitude, latitude):
    """Allow to retrieve a depth value from EMODnet web services"""
    logger = log.logging.getLogger(__file__)
    # r = requests.get(f"https://rest.emodnet-bathymetry.eu/depth_sample?geom=POINT({longitude} {latitude}) ")
    r = requests.get(
        f"https://www.gmrt.org/services/PointServer?latitude={latitude}&longitude={longitude}&format=geojson) ",
        timeout=10,
    )

    if r.status_code != 200:
        logger.error(f"Error while querying data {r.status_code} : {r.reason}")
        return float("nan")
    data = json.loads(r.text)
    return float(data)


def retrieve_depth(longitude, latitude):
    return retrieve_depth_gmrt(longitude, latitude)
